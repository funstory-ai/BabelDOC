use lopdf::{Document, Object};

#[cfg(not(feature = "async"))]
use lopdf::LoadOptions;

#[cfg(not(feature = "async"))]
#[test]
fn test_load_encrypted_pdf_from_assets() {
    // Test loading the existing encrypted.pdf file
    let doc = Document::load("assets/encrypted.pdf").unwrap();

    assert!(
        !doc.is_encrypted(),
        "Document should not appear encrypted after decryption"
    );
    assert!(doc.encryption_state.is_some());

    // Check that we can access the decrypted content
    let pages = doc.get_pages();
    assert_eq!(pages.len(), 1, "Should have exactly one page");

    // Try to extract text from the document
    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    // The document should be readable even if encrypted with empty password
    let text = doc.extract_text(&page_numbers).unwrap();

    // Verify we can extract meaningful text
    assert!(text.contains("USCIS"), "Should contain USCIS text from the form");
    assert!(text.contains("Form G-1145"), "Should contain form number");

    // Verify we can access objects
    for i in 1..=10 {
        // Should be able to access at least the first 10 objects
        assert!(
            doc.get_object((i, 0)).is_ok(),
            "Should be able to access object ({}, 0)",
            i
        );
    }

    // Verify trailer has required entries
    assert!(doc.trailer.get(b"Root").is_ok(), "Trailer should have Root entry");
    assert!(
        doc.trailer.get(b"Encrypt").is_err(),
        "Encrypt entry should be removed after decryption"
    );
    assert!(doc.trailer.get(b"Info").is_ok(), "Trailer should have Info entry");
}

#[cfg(not(feature = "async"))]
#[test]
fn test_decrypt_pdf_with_empty_password() {
    // Create a simple PDF document
    let mut doc = Document::with_version("1.5");

    // Add an ID to the trailer (required for encryption)
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Create a simple page structure
    let pages_id = doc.new_object_id();
    let page_id = doc.new_object_id();
    let content_id = doc.new_object_id();
    let font_id = doc.new_object_id();
    let resources_id = doc.new_object_id();

    // Create catalog
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Create pages
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page_id)],
        "Count" => 1
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create resources
    let resources_dict = lopdf::dictionary! {
        "Font" => lopdf::dictionary! {
            "F1" => Object::Reference(font_id)
        }
    };
    doc.objects.insert(resources_id, Object::Dictionary(resources_dict));

    // Create page
    let page_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => Object::Reference(resources_id),
        "Contents" => Object::Reference(content_id)
    };
    doc.objects.insert(page_id, Object::Dictionary(page_dict));

    // Create font
    let font_dict = lopdf::dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    };
    doc.objects.insert(font_id, Object::Dictionary(font_dict));

    // Create content stream
    let content = b"BT\n/F1 12 Tf\n100 700 Td\n(Hello, Encrypted World!) Tj\nET\n";
    let content_stream = lopdf::Stream::new(lopdf::dictionary! {}, content.to_vec());
    doc.objects.insert(content_id, Object::Stream(content_stream));

    // Save to a temporary file
    let temp_dir = tempfile::tempdir().unwrap();
    let unencrypted_path = temp_dir.path().join("test_unencrypted.pdf");
    doc.save(&unencrypted_path).unwrap();

    // Encrypt the document with empty password
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "",
        user_password: "",
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save encrypted document
    let encrypted_path = temp_dir.path().join("test_encrypted.pdf");
    doc.save(&encrypted_path).unwrap();

    // Now test loading the encrypted document
    let loaded_doc = Document::load(&encrypted_path).unwrap();

    assert!(
        !loaded_doc.is_encrypted(),
        "Should not appear encrypted after decryption"
    );
    assert!(loaded_doc.encryption_state.is_some());

    // Check that we can access the decrypted content
    let pages = loaded_doc.get_pages();
    assert_eq!(pages.len(), 1);

    // Extract text to verify decryption worked
    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    let text = loaded_doc.extract_text(&page_numbers).unwrap();
    assert!(text.contains("Hello, Encrypted World!"));
}

#[cfg(not(feature = "async"))]
#[test]
#[ignore] // Object streams with encryption need more work
fn test_decrypt_pdf_with_object_streams() {
    // Create a document with object streams
    let mut doc = Document::with_version("1.5");

    // Add an ID to the trailer
    let id1 = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    let id2 = vec![160u8, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Create catalog
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0))
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Create pages tree
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference((3, 0))],
        "Count" => 1
    };
    doc.objects.insert((2, 0), Object::Dictionary(pages_dict));

    // Create page
    let page_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference((2, 0)),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => lopdf::dictionary! {
            "Font" => lopdf::dictionary! {
                "F1" => Object::Reference((4, 0))
            }
        },
        "Contents" => Object::Reference((5, 0))
    };
    doc.objects.insert((3, 0), Object::Dictionary(page_dict));

    // Create font
    let font_dict = lopdf::dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    };
    doc.objects.insert((4, 0), Object::Dictionary(font_dict));

    // Create content stream
    let content = b"BT\n/F1 12 Tf\n100 700 Td\n(Test with Object Streams!) Tj\nET\n";
    let content_stream = lopdf::Stream::new(lopdf::dictionary! {}, content.to_vec());
    doc.objects.insert((5, 0), Object::Stream(content_stream));

    // Compress document using object streams
    doc.compress();

    // Encrypt the document
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "owner",
        user_password: "",
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save encrypted document
    let temp_dir = tempfile::tempdir().unwrap();
    let encrypted_path = temp_dir.path().join("test_encrypted_objstream.pdf");
    doc.save(&encrypted_path).unwrap();

    // Load and verify
    let loaded_doc = Document::load(&encrypted_path).unwrap();
    assert!(loaded_doc.is_encrypted());

    // Verify we can access the content
    let pages = loaded_doc.get_pages();
    assert_eq!(pages.len(), 1);

    // Extract text to verify decryption worked
    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    let text = loaded_doc.extract_text(&page_numbers).unwrap();
    assert!(text.contains("Test with Object Streams!"));
}

#[cfg(not(feature = "async"))]
#[test]
#[ignore] // Raw object extraction needs adjustments for new decryption approach
fn test_encrypted_pdf_raw_object_extraction() {
    // This test verifies that the raw object extraction works correctly
    // for encrypted PDFs, which is crucial for the pdftk-style decryption

    let mut doc = Document::with_version("1.5");

    // Add ID
    let id1 = vec![99u8; 16];
    let id2 = vec![88u8; 16];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Create a minimal document structure
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0))
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Add pages tree
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![],
        "Count" => 0
    };
    doc.objects.insert((2, 0), Object::Dictionary(pages_dict));

    // Add some test objects with different types
    doc.objects.insert((10, 0), Object::Integer(42));
    doc.objects.insert(
        (11, 0),
        Object::String(b"test string".to_vec(), lopdf::StringFormat::Literal),
    );
    doc.objects.insert(
        (12, 0),
        Object::Array(vec![Object::Integer(1), Object::Integer(2), Object::Integer(3)]),
    );

    // Encrypt
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "test",
        user_password: "",
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save and reload
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("test_raw_extraction.pdf");
    doc.save(&path).unwrap();

    let loaded_doc = Document::load(&path).unwrap();
    assert!(loaded_doc.is_encrypted());

    // Verify that all objects were properly decrypted
    assert_eq!(loaded_doc.get_object((10, 0)).unwrap().as_i64().unwrap(), 42);

    let string_obj = loaded_doc.get_object((11, 0)).unwrap();
    if let Object::String(bytes, _) = string_obj {
        assert_eq!(bytes, b"test string");
    } else {
        panic!("Expected string object");
    }

    let array_obj = loaded_doc.get_object((12, 0)).unwrap();
    if let Object::Array(arr) = array_obj {
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_i64().unwrap(), 1);
        assert_eq!(arr[1].as_i64().unwrap(), 2);
        assert_eq!(arr[2].as_i64().unwrap(), 3);
    } else {
        panic!("Expected array object");
    }
}

#[cfg(not(feature = "async"))]
#[test]
#[ignore] // Structure preservation test needs adjustments
fn test_encrypted_pdf_preserves_structure() {
    // Test that the document structure is preserved after encryption/decryption
    let mut doc = Document::with_version("1.5");

    // Add ID
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(vec![77u8; 16], lopdf::StringFormat::Literal),
            Object::String(vec![66u8; 16], lopdf::StringFormat::Literal),
        ]),
    );

    // Create a complex structure
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0)),
        "Metadata" => Object::Reference((3, 0))
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Pages tree
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference((4, 0))],
        "Count" => 1
    };
    doc.objects.insert((2, 0), Object::Dictionary(pages_dict));

    // Metadata stream
    let metadata = b"<rdf:RDF>test metadata</rdf:RDF>";
    let metadata_stream = lopdf::Stream::new(
        lopdf::dictionary! {
            "Type" => "Metadata",
            "Subtype" => "XML"
        },
        metadata.to_vec(),
    );
    doc.objects.insert((3, 0), Object::Stream(metadata_stream));

    // Page
    let page_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference((2, 0)),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => lopdf::dictionary! {}
    };
    doc.objects.insert((4, 0), Object::Dictionary(page_dict));

    // Encrypt
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "complex",
        user_password: "",
        key_length: 128,
        permissions: lopdf::Permissions::all(),
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save and reload
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("test_structure.pdf");
    doc.save(&path).unwrap();

    let loaded_doc = Document::load(&path).unwrap();
    assert!(loaded_doc.is_encrypted());

    // Verify structure is preserved
    let root = loaded_doc.trailer.get(b"Root").unwrap().as_reference().unwrap();
    let catalog = loaded_doc.get_object(root).unwrap();

    if let Object::Dictionary(dict) = catalog {
        assert_eq!(dict.get(b"Type").unwrap(), &Object::Name(b"Catalog".to_vec()));
        assert!(dict.has(b"Pages"));
        assert!(dict.has(b"Metadata"));
    } else {
        panic!("Expected catalog to be a dictionary");
    }

    // Check metadata stream was decrypted correctly
    let metadata_obj = loaded_doc.get_object((3, 0)).unwrap();
    if let Object::Stream(stream) = metadata_obj {
        assert_eq!(stream.dict.get(b"Type").unwrap(), &Object::Name(b"Metadata".to_vec()));
        // Note: Content might be compressed, so we just check it exists
        assert!(!stream.content.is_empty());
    } else {
        panic!("Expected metadata to be a stream");
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_decrypt_pdf_with_empty_password_async() {
    // Create a simple PDF document
    let mut doc = Document::with_version("1.5");

    // Add an ID to the trailer (required for encryption)
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Create a simple page structure (similar to sync version)
    let pages_id = doc.new_object_id();
    let page_id = doc.new_object_id();
    let content_id = doc.new_object_id();
    let font_id = doc.new_object_id();
    let resources_id = doc.new_object_id();

    // Create catalog
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Create pages
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page_id)],
        "Count" => 1
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create resources
    let resources_dict = lopdf::dictionary! {
        "Font" => lopdf::dictionary! {
            "F1" => Object::Reference(font_id)
        }
    };
    doc.objects.insert(resources_id, Object::Dictionary(resources_dict));

    // Create page
    let page_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => Object::Reference(resources_id),
        "Contents" => Object::Reference(content_id)
    };
    doc.objects.insert(page_id, Object::Dictionary(page_dict));

    // Create font
    let font_dict = lopdf::dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    };
    doc.objects.insert(font_id, Object::Dictionary(font_dict));

    // Create content stream
    let content = b"BT\n/F1 12 Tf\n100 700 Td\n(Hello, Async Encrypted World!) Tj\nET\n";
    let content_stream = lopdf::Stream::new(lopdf::dictionary! {}, content.to_vec());
    doc.objects.insert(content_id, Object::Stream(content_stream));

    // Save to a temporary file
    let temp_dir = tempfile::tempdir().unwrap();
    let encrypted_path = temp_dir.path().join("test_encrypted_async.pdf");

    // Encrypt the document with empty password
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "",
        user_password: "",
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();
    doc.save(&encrypted_path).unwrap();

    // Now test loading the encrypted document asynchronously
    let loaded_doc = Document::load(&encrypted_path).await.unwrap();

    assert!(
        !loaded_doc.is_encrypted(),
        "Should not appear encrypted after decryption"
    );
    assert!(loaded_doc.encryption_state.is_some());

    let pages = loaded_doc.get_pages();
    assert_eq!(pages.len(), 1);

    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    let text = loaded_doc.extract_text(&page_numbers).unwrap();
    assert!(text.contains("Hello, Async Encrypted World!"));
}

#[cfg(not(feature = "async"))]
#[test]
fn test_load_with_password_correct_password() {
    // Create a simple PDF document
    let mut doc = Document::with_version("1.5");

    // Add an ID to the trailer (required for encryption)
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Create a simple page structure
    let pages_id = doc.new_object_id();
    let page_id = doc.new_object_id();
    let content_id = doc.new_object_id();
    let font_id = doc.new_object_id();
    let resources_id = doc.new_object_id();

    // Create catalog
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Create pages
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page_id)],
        "Count" => 1
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create resources
    let resources_dict = lopdf::dictionary! {
        "Font" => lopdf::dictionary! {
            "F1" => Object::Reference(font_id)
        }
    };
    doc.objects.insert(resources_id, Object::Dictionary(resources_dict));

    // Create page
    let page_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => Object::Reference(resources_id),
        "Contents" => Object::Reference(content_id)
    };
    doc.objects.insert(page_id, Object::Dictionary(page_dict));

    // Create font
    let font_dict = lopdf::dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    };
    doc.objects.insert(font_id, Object::Dictionary(font_dict));

    // Create content stream
    let content = b"BT\n/F1 12 Tf\n100 700 Td\n(Password Protected Content!) Tj\nET\n";
    let content_stream = lopdf::Stream::new(lopdf::dictionary! {}, content.to_vec());
    doc.objects.insert(content_id, Object::Stream(content_stream));

    // Encrypt the document with a NON-EMPTY password
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "owner_secret",
        user_password: "user_secret", // Non-empty password!
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save encrypted document
    let temp_dir = tempfile::tempdir().unwrap();
    let encrypted_path = temp_dir.path().join("test_password_protected.pdf");
    doc.save(&encrypted_path).unwrap();

    // Test 1: Regular load() should fail to decrypt (no objects loaded because empty password doesn't work)
    let loaded_without_password = Document::load(&encrypted_path).unwrap();
    assert!(
        loaded_without_password.is_encrypted(),
        "Should still appear encrypted when auth fails"
    );

    // load_with_password() with correct password should work
    let loaded_with_password = Document::load_with_password(&encrypted_path, "user_secret").unwrap();
    assert!(
        !loaded_with_password.is_encrypted(),
        "Should not appear encrypted after successful decryption"
    );
    assert!(loaded_with_password.encryption_state.is_some());

    let pages = loaded_with_password.get_pages();
    assert_eq!(pages.len(), 1, "Should have exactly one page");

    // Extract text to verify decryption worked
    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    let text = loaded_with_password.extract_text(&page_numbers).unwrap();
    assert!(
        text.contains("Password Protected Content!"),
        "Should be able to extract text: {}",
        text
    );
}

#[cfg(not(feature = "async"))]
#[test]
fn test_load_with_password_wrong_password() {
    use lopdf::Error;

    // Create a simple PDF document
    let mut doc = Document::with_version("1.5");

    // Add an ID to the trailer
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Minimal document structure
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0))
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![],
        "Count" => 0
    };
    doc.objects.insert((2, 0), Object::Dictionary(pages_dict));

    // Encrypt with a specific password
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "correct_owner",
        user_password: "correct_user",
        key_length: 128,
        permissions: lopdf::Permissions::all(),
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("test_wrong_password.pdf");
    doc.save(&path).unwrap();

    // Try to load with wrong password - should fail
    let result = Document::load_with_password(&path, "wrong_password");
    assert!(result.is_err(), "Should fail with wrong password");

    if let Err(Error::InvalidPassword) = result {
        // Good - got the expected error
    } else {
        panic!("Expected InvalidPassword error, got: {:?}", result);
    }
}

#[cfg(not(feature = "async"))]
#[test]
fn test_load_with_password_empty_password_when_required() {
    use lopdf::Error;

    // Create a simple PDF document
    let mut doc = Document::with_version("1.5");

    // Add ID
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Minimal document
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0))
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![],
        "Count" => 0
    };
    doc.objects.insert((2, 0), Object::Dictionary(pages_dict));

    // Encrypt with NON-empty password
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "secret",
        user_password: "secret",
        key_length: 128,
        permissions: lopdf::Permissions::all(),
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("test_empty_password.pdf");
    doc.save(&path).unwrap();

    // Try to load with empty password when document requires a real password
    // This should fail with InvalidPassword
    let result = Document::load_with_password(&path, "");
    assert!(result.is_err(), "Should fail when empty password doesn't work");

    if let Err(Error::InvalidPassword) = result {
        // Good - got the expected error
    } else {
        panic!("Expected InvalidPassword error, got: {:?}", result);
    }
}

#[cfg(not(feature = "async"))]
#[test]
fn test_load_mem_with_password() {
    // Create a simple PDF document
    let mut doc = Document::with_version("1.5");

    // Add ID
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Create a simple page structure
    let pages_id = doc.new_object_id();
    let page_id = doc.new_object_id();
    let content_id = doc.new_object_id();

    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page_id)],
        "Count" => 1
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    let page_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Contents" => Object::Reference(content_id)
    };
    doc.objects.insert(page_id, Object::Dictionary(page_dict));

    let content = b"BT\n/F1 12 Tf\n100 700 Td\n(Memory Loaded!) Tj\nET\n";
    let content_stream = lopdf::Stream::new(lopdf::dictionary! {}, content.to_vec());
    doc.objects.insert(content_id, Object::Stream(content_stream));

    // Encrypt with password
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "mem_owner",
        user_password: "mem_user",
        key_length: 128,
        permissions: lopdf::Permissions::all(),
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save to memory buffer
    let mut buffer = Vec::new();
    doc.save_to(&mut buffer).unwrap();

    let loaded_doc = Document::load_mem_with_options(&buffer, LoadOptions::with_password("mem_user")).unwrap();
    assert!(
        !loaded_doc.is_encrypted(),
        "Should not appear encrypted after decryption"
    );
    assert!(loaded_doc.encryption_state.is_some());

    let pages = loaded_doc.get_pages();
    assert_eq!(pages.len(), 1);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_load_with_password_async() {
    // Create a simple PDF document
    let mut doc = Document::with_version("1.5");

    // Add ID
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Create minimal structure
    let pages_id = doc.new_object_id();
    let page_id = doc.new_object_id();
    let content_id = doc.new_object_id();

    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page_id)],
        "Count" => 1
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    let page_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Contents" => Object::Reference(content_id)
    };
    doc.objects.insert(page_id, Object::Dictionary(page_dict));

    let content = b"BT\n/F1 12 Tf\n100 700 Td\n(Async Password Protected!) Tj\nET\n";
    let content_stream = lopdf::Stream::new(lopdf::dictionary! {}, content.to_vec());
    doc.objects.insert(content_id, Object::Stream(content_stream));

    // Encrypt
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "async_owner",
        user_password: "async_user",
        key_length: 128,
        permissions: lopdf::Permissions::all(),
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("test_async_password.pdf");
    doc.save(&path).unwrap();

    let loaded_doc = Document::load_with_password(&path, "async_user").await.unwrap();
    assert!(
        !loaded_doc.is_encrypted(),
        "Should not appear encrypted after decryption"
    );
    assert!(loaded_doc.encryption_state.is_some());

    let pages = loaded_doc.get_pages();
    assert_eq!(pages.len(), 1);
}
#[cfg(not(feature = "async"))]
#[test]
fn test_load_with_password_multipage_pdf() {
    // Create a PDF document with multiple pages
    let mut doc = Document::with_version("1.5");

    // Add an ID to the trailer (required for encryption)
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    // Create a multi-page document
    let pages_id = doc.new_object_id();
    let page1_id = doc.new_object_id();
    let page2_id = doc.new_object_id();
    let page3_id = doc.new_object_id();
    let content1_id = doc.new_object_id();
    let content2_id = doc.new_object_id();
    let content3_id = doc.new_object_id();
    let font_id = doc.new_object_id();
    let resources_id = doc.new_object_id();

    // Create catalog
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Create pages tree with 3 pages
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page1_id), Object::Reference(page2_id), Object::Reference(page3_id)],
        "Count" => 3
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create resources
    let resources_dict = lopdf::dictionary! {
        "Font" => lopdf::dictionary! {
            "F1" => Object::Reference(font_id)
        }
    };
    doc.objects.insert(resources_id, Object::Dictionary(resources_dict));

    // Create page 1
    let page1_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => Object::Reference(resources_id),
        "Contents" => Object::Reference(content1_id)
    };
    doc.objects.insert(page1_id, Object::Dictionary(page1_dict));

    // Create page 2
    let page2_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => Object::Reference(resources_id),
        "Contents" => Object::Reference(content2_id)
    };
    doc.objects.insert(page2_id, Object::Dictionary(page2_dict));

    // Create page 3
    let page3_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => Object::Reference(resources_id),
        "Contents" => Object::Reference(content3_id)
    };
    doc.objects.insert(page3_id, Object::Dictionary(page3_dict));

    // Create font
    let font_dict = lopdf::dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    };
    doc.objects.insert(font_id, Object::Dictionary(font_dict));

    // Create content streams
    let content1 = b"BT\n/F1 12 Tf\n100 700 Td\n(Page 1 Content!) Tj\nET\n";
    let content1_stream = lopdf::Stream::new(lopdf::dictionary! {}, content1.to_vec());
    doc.objects.insert(content1_id, Object::Stream(content1_stream));

    let content2 = b"BT\n/F1 12 Tf\n100 700 Td\n(Page 2 Content!) Tj\nET\n";
    let content2_stream = lopdf::Stream::new(lopdf::dictionary! {}, content2.to_vec());
    doc.objects.insert(content2_id, Object::Stream(content2_stream));

    let content3 = b"BT\n/F1 12 Tf\n100 700 Td\n(Page 3 Content!) Tj\nET\n";
    let content3_stream = lopdf::Stream::new(lopdf::dictionary! {}, content3.to_vec());
    doc.objects.insert(content3_id, Object::Stream(content3_stream));

    // Encrypt the document with a NON-EMPTY password
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "owner_secret",
        user_password: "user_secret", // Non-empty password!
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save encrypted document
    let temp_dir = tempfile::tempdir().unwrap();
    let encrypted_path = temp_dir.path().join("test_multipage_encrypted.pdf");
    doc.save(&encrypted_path).unwrap();

    // Get file size for debugging
    let file_metadata = std::fs::metadata(&encrypted_path).unwrap();
    let original_size = file_metadata.len();
    println!("Encrypted PDF size: {} bytes", original_size);

    let loaded_doc = Document::load_with_password(&encrypted_path, "user_secret").unwrap();
    assert!(
        !loaded_doc.is_encrypted(),
        "Should not appear encrypted after decryption"
    );
    assert!(loaded_doc.encryption_state.is_some());

    let pages = loaded_doc.get_pages();
    assert_eq!(pages.len(), 3, "Should have exactly 3 pages, but got {}", pages.len());

    let object_count = loaded_doc.objects.len();
    println!("Loaded document has {} objects", object_count);

    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    let text = loaded_doc.extract_text(&page_numbers).unwrap();
    println!("Extracted text: {}", text);

    assert!(
        text.contains("Page 1 Content!"),
        "Should contain Page 1 content: {}",
        text
    );
    assert!(
        text.contains("Page 2 Content!"),
        "Should contain Page 2 content: {}",
        text
    );
    assert!(
        text.contains("Page 3 Content!"),
        "Should contain Page 3 content: {}",
        text
    );
}

#[cfg(not(feature = "async"))]
#[test]
fn test_load_with_password_with_compressed_streams() {
    // Create a PDF document with compressed streams
    let mut doc = Document::with_version("1.5");

    // Add an ID to the trailer (required for encryption)
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    let pages_id = doc.new_object_id();
    let page1_id = doc.new_object_id();
    let page2_id = doc.new_object_id();
    let content1_id = doc.new_object_id();
    let content2_id = doc.new_object_id();
    let font_id = doc.new_object_id();
    let resources_id = doc.new_object_id();

    // Create catalog
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Create pages tree with 2 pages
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page1_id), Object::Reference(page2_id)],
        "Count" => 2
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create resources
    let resources_dict = lopdf::dictionary! {
        "Font" => lopdf::dictionary! {
            "F1" => Object::Reference(font_id)
        }
    };
    doc.objects.insert(resources_id, Object::Dictionary(resources_dict));

    // Create page 1
    let page1_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => Object::Reference(resources_id),
        "Contents" => Object::Reference(content1_id)
    };
    doc.objects.insert(page1_id, Object::Dictionary(page1_dict));

    // Create page 2
    let page2_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Resources" => Object::Reference(resources_id),
        "Contents" => Object::Reference(content2_id)
    };
    doc.objects.insert(page2_id, Object::Dictionary(page2_dict));

    // Create font
    let font_dict = lopdf::dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    };
    doc.objects.insert(font_id, Object::Dictionary(font_dict));

    // Create content streams - Note: these will be compressed by the Stream
    let content1 = b"BT\n/F1 12 Tf\n100 700 Td\n(Compressed Page 1!) Tj\nET\n";
    let mut content1_stream = lopdf::Stream::new(lopdf::dictionary! { "Filter" => "FlateDecode" }, content1.to_vec());
    content1_stream.compress().unwrap();
    doc.objects.insert(content1_id, Object::Stream(content1_stream));

    let content2 = b"BT\n/F1 12 Tf\n100 700 Td\n(Compressed Page 2!) Tj\nET\n";
    let mut content2_stream = lopdf::Stream::new(lopdf::dictionary! { "Filter" => "FlateDecode" }, content2.to_vec());
    content2_stream.compress().unwrap();
    doc.objects.insert(content2_id, Object::Stream(content2_stream));

    // Encrypt the document
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "owner_secret",
        user_password: "user_secret",
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save encrypted document
    let temp_dir = tempfile::tempdir().unwrap();
    let encrypted_path = temp_dir.path().join("test_compressed_encrypted.pdf");
    doc.save(&encrypted_path).unwrap();

    // Get file size for debugging
    let file_metadata = std::fs::metadata(&encrypted_path).unwrap();
    println!("Compressed encrypted PDF size: {} bytes", file_metadata.len());

    let loaded_doc = Document::load_with_password(&encrypted_path, "user_secret").unwrap();
    assert!(
        !loaded_doc.is_encrypted(),
        "Should not appear encrypted after decryption"
    );
    assert!(loaded_doc.encryption_state.is_some());

    let pages = loaded_doc.get_pages();
    println!("Loaded {} pages", pages.len());
    assert_eq!(pages.len(), 2, "Should have exactly 2 pages, but got {}", pages.len());

    let object_count = loaded_doc.objects.len();
    println!("Loaded document has {} objects", object_count);
}

#[cfg(not(feature = "async"))]
#[test]
fn test_load_with_password_stream_with_endobj_bytes() {
    // Test with binary content that contains "endobj" substring
    let mut doc = Document::with_version("1.5");

    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    let pages_id = doc.new_object_id();
    let page_id = doc.new_object_id();
    let content_id = doc.new_object_id();
    let second_obj_id = doc.new_object_id();

    // Create catalog
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Create pages tree
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page_id)],
        "Count" => 1
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create page
    let page_dict = lopdf::dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
        "Contents" => Object::Reference(content_id)
    };
    doc.objects.insert(page_id, Object::Dictionary(page_dict));

    // Create a content stream that contains "endobj" in its binary content
    // This simulates what can happen with compressed/binary streams
    let mut content = b"BT\n/F1 12 Tf\n100 700 Td\n(Test) Tj\nET\n".to_vec();
    // Add "endobj" bytes in the stream content
    content.extend_from_slice(b"endobj fake marker");
    let content_stream = lopdf::Stream::new(lopdf::dictionary! {}, content);
    doc.objects.insert(content_id, Object::Stream(content_stream));

    // Add another object that should be loaded after the stream
    doc.objects.insert(second_obj_id, Object::Integer(42));

    // Encrypt the document
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "owner_secret",
        user_password: "user_secret",
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save encrypted document
    let temp_dir = tempfile::tempdir().unwrap();
    let encrypted_path = temp_dir.path().join("test_stream_with_endobj.pdf");
    doc.save(&encrypted_path).unwrap();

    println!("PDF with endobj in stream saved to: {}", encrypted_path.display());

    // Load with password
    let loaded_doc = Document::load_with_password(&encrypted_path, "user_secret").unwrap();

    // Verify all objects were loaded
    let object_count = loaded_doc.objects.len();
    println!("Loaded {} objects", object_count);

    // The second object should be loaded
    assert!(
        loaded_doc.get_object(second_obj_id).is_ok(),
        "Second object should be loaded"
    );
}

#[cfg(not(feature = "async"))]
#[test]
fn test_load_encrypted_pdf_with_object_streams() {
    // Load the encrypted.pdf from assets
    let doc = Document::load("assets/encrypted.pdf").unwrap();

    println!("Document version: {}", doc.version);
    println!("Number of objects: {}", doc.objects.len());

    // Check if document has object streams
    let mut has_obj_stream = false;
    for (id, obj) in &doc.objects {
        if let Object::Stream(stream) = obj
            && stream.dict.has_type(b"ObjStm")
        {
            has_obj_stream = true;
            println!("Found object stream: {:?}", id);
        }
    }
    println!("Has object streams: {}", has_obj_stream);

    // Check the pages
    let pages = doc.get_pages();
    println!("Number of pages: {}", pages.len());

    // Try to extract text
    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    let text = doc.extract_text(&page_numbers).unwrap();
    println!("Extracted {} characters of text", text.len());

    assert!(!pages.is_empty(), "Should have at least one page");

    // Now save and reload to verify round-trip
    let temp_dir = tempfile::tempdir().unwrap();
    let saved_path = temp_dir.path().join("encrypted_resaved.pdf");

    let mut doc_clone = doc.clone();
    doc_clone.save(&saved_path).unwrap();

    let file_size = std::fs::metadata(&saved_path).unwrap().len();
    println!("Saved file size: {} bytes", file_size);

    // Verify the saved file can be loaded and has the same content
    let reloaded = Document::load(&saved_path).unwrap();
    let reloaded_pages = reloaded.get_pages();
    println!(
        "Reloaded document has {} pages and {} objects",
        reloaded_pages.len(),
        reloaded.objects.len()
    );

    assert_eq!(
        pages.len(),
        reloaded_pages.len(),
        "Should have same number of pages after round-trip"
    );
}

#[cfg(not(feature = "async"))]
#[test]
fn test_encrypt_decrypt_multipage_roundtrip() {
    // Create a PDF document with multiple pages
    let mut doc = Document::with_version("1.5");

    // Add an ID to the trailer
    let id1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id2 = vec![16u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1.clone(), lopdf::StringFormat::Literal),
            Object::String(id2.clone(), lopdf::StringFormat::Literal),
        ]),
    );

    // Create a multi-page document with 5 pages
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();
    let mut content_ids = Vec::new();

    for _ in 0..5 {
        page_ids.push(doc.new_object_id());
        content_ids.push(doc.new_object_id());
    }

    let font_id = doc.new_object_id();
    let resources_id = doc.new_object_id();

    // Create catalog
    let catalog_dict = lopdf::dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id)
    };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Create pages tree with 5 pages
    let pages_dict = lopdf::dictionary! {
        "Type" => "Pages",
        "Kids" => page_ids.iter().map(|id| Object::Reference(*id)).collect::<Vec<_>>(),
        "Count" => 5
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create resources
    let resources_dict = lopdf::dictionary! {
        "Font" => lopdf::dictionary! {
            "F1" => Object::Reference(font_id)
        }
    };
    doc.objects.insert(resources_id, Object::Dictionary(resources_dict));

    // Create pages and content
    for (i, (page_id, content_id)) in page_ids.iter().zip(content_ids.iter()).enumerate() {
        // Create page
        let page_dict = lopdf::dictionary! {
            "Type" => "Page",
            "Parent" => Object::Reference(pages_id),
            "MediaBox" => vec![Object::Integer(0), Object::Integer(0), Object::Integer(612), Object::Integer(792)],
            "Resources" => Object::Reference(resources_id),
            "Contents" => Object::Reference(*content_id)
        };
        doc.objects.insert(*page_id, Object::Dictionary(page_dict));

        // Create content stream
        let content = format!(
            "BT\n/F1 12 Tf\n100 700 Td\n(Page {} Content - Test String!) Tj\nET\n",
            i + 1
        );
        let content_stream = lopdf::Stream::new(lopdf::dictionary! {}, content.into_bytes());
        doc.objects.insert(*content_id, Object::Stream(content_stream));
    }

    // Create font
    let font_dict = lopdf::dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    };
    doc.objects.insert(font_id, Object::Dictionary(font_dict));

    // Save unencrypted version first for size comparison
    let temp_dir = tempfile::tempdir().unwrap();
    let unencrypted_path = temp_dir.path().join("multipage_unencrypted.pdf");
    doc.save(&unencrypted_path).unwrap();
    let unencrypted_size = std::fs::metadata(&unencrypted_path).unwrap().len();
    println!("Unencrypted PDF size: {} bytes", unencrypted_size);

    // Encrypt the document with a NON-EMPTY password
    let permissions = lopdf::Permissions::all();
    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "owner_password",
        user_password: "test_password",
        key_length: 128,
        permissions,
    };

    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // Save encrypted document
    let encrypted_path = temp_dir.path().join("multipage_encrypted.pdf");
    doc.save(&encrypted_path).unwrap();
    let encrypted_size = std::fs::metadata(&encrypted_path).unwrap().len();
    println!("Encrypted PDF size: {} bytes", encrypted_size);

    // Now load the encrypted PDF with password
    let loaded_doc = Document::load_with_password(&encrypted_path, "test_password").unwrap();

    let loaded_pages = loaded_doc.get_pages();
    let loaded_objects = loaded_doc.objects.len();
    println!(
        "Loaded document: {} pages, {} objects",
        loaded_pages.len(),
        loaded_objects
    );

    // Verify all 5 pages are loaded
    assert_eq!(loaded_pages.len(), 5, "Should have 5 pages, got {}", loaded_pages.len());

    // Extract text from all pages
    let page_numbers: Vec<u32> = loaded_pages.keys().cloned().collect();
    let text = loaded_doc.extract_text(&page_numbers).unwrap();
    println!("Extracted text length: {} chars", text.len());

    // Verify content from each page
    for i in 1..=5 {
        let expected = format!("Page {}", i);
        assert!(
            text.contains(&expected),
            "Should contain text from page {}: text = '{}'",
            i,
            text
        );
    }

    // Re-save the loaded document (this is the user's scenario)
    let resaved_path = temp_dir.path().join("multipage_resaved.pdf");
    let mut loaded_doc_mut = loaded_doc.clone();
    loaded_doc_mut.save(&resaved_path).unwrap();
    let resaved_size = std::fs::metadata(&resaved_path).unwrap().len();
    println!("Re-saved PDF size: {} bytes", resaved_size);

    // The re-saved file should be similar size (within reasonable bounds)
    // It shouldn't be drastically smaller like the user's issue (468 bytes vs 197KB)
    assert!(
        resaved_size > unencrypted_size / 2,
        "Re-saved file is too small! Got {} bytes, expected at least {} bytes",
        resaved_size,
        unencrypted_size / 2
    );

    // Load the re-saved document and verify pages
    let reloaded = Document::load(&resaved_path).unwrap();
    let reloaded_pages = reloaded.get_pages();
    println!(
        "Re-loaded document: {} pages, {} objects",
        reloaded_pages.len(),
        reloaded.objects.len()
    );

    assert_eq!(
        reloaded_pages.len(),
        5,
        "Re-loaded should have 5 pages, got {}",
        reloaded_pages.len()
    );
}

#[cfg(not(feature = "async"))]
#[test]
fn test_was_encrypted_method() {
    // Test 1: Unencrypted document
    let mut doc = Document::with_version("1.5");
    let catalog_dict = lopdf::dictionary! { "Type" => "Catalog" };
    let catalog_id = doc.add_object(catalog_dict);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    assert!(!doc.is_encrypted(), "Unencrypted doc should not be encrypted");
    assert!(!doc.was_encrypted(), "Unencrypted doc was not originally encrypted");

    // Test 2: Create and load encrypted document
    let id1 = vec![1u8; 16];
    let id2 = vec![2u8; 16];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id1, lopdf::StringFormat::Literal),
            Object::String(id2, lopdf::StringFormat::Literal),
        ]),
    );

    let encryption_version = lopdf::EncryptionVersion::V2 {
        document: &doc,
        owner_password: "owner",
        user_password: "user",
        key_length: 128,
        permissions: lopdf::Permissions::all(),
    };
    let encryption_state = lopdf::EncryptionState::try_from(encryption_version).unwrap();
    doc.encrypt(&encryption_state).unwrap();

    // After encryption, before saving
    assert!(doc.is_encrypted(), "Should be encrypted after encrypt()");

    // Save and reload with password
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("test_was_encrypted.pdf");
    doc.save(&path).unwrap();

    let loaded = Document::load_with_password(&path, "user").unwrap();

    // After loading with correct password: decrypted but was_encrypted is true
    assert!(!loaded.is_encrypted(), "Should not appear encrypted after decryption");
    assert!(loaded.was_encrypted(), "Should remember it was originally encrypted");

    // Test 3: Load encrypted doc without correct password (empty doesn't work)
    let loaded_locked = Document::load(&path).unwrap();
    assert!(
        loaded_locked.is_encrypted(),
        "Should still appear encrypted without password"
    );
    assert!(
        !loaded_locked.was_encrypted(),
        "encryption_state not set when auth failed"
    );
}
