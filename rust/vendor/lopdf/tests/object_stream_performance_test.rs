use lopdf::{Document, Object, ObjectStream, SaveOptions, dictionary};
use std::time::Instant;

#[test]
fn test_can_be_compressed_performance() {
    // Create a document with many objects
    let mut doc = Document::with_version("1.5");
    let mut object_ids = Vec::new();

    // Create 1000 objects
    for i in 0..1000 {
        let obj_id = doc.add_object(dictionary! {
            "Type" => format!("TestObject{}", i),
            "Index" => i as i64,
            "Data" => format!("This is test object number {}", i)
        });
        object_ids.push(obj_id);
    }

    // Add some to trailer
    doc.trailer.set("Root", object_ids[0]);
    doc.trailer.set("Info", object_ids[1]);
    doc.trailer.set("Custom1", object_ids[2]);
    doc.trailer.set("Custom2", object_ids[3]);

    // Measure performance of can_be_compressed checks
    let start = Instant::now();
    let mut compressible_count = 0;

    for &id in &object_ids {
        if let Some(obj) = doc.objects.get(&id)
            && ObjectStream::can_be_compressed(id, obj, &doc)
        {
            compressible_count += 1;
        }
    }

    let duration = start.elapsed();

    println!("Checked {} objects in {:?}", object_ids.len(), duration);
    println!("Compressible objects: {}", compressible_count);

    // All objects should be compressible (none are encryption dicts)
    assert_eq!(
        compressible_count,
        object_ids.len(),
        "All non-encryption objects should be compressible"
    );

    // Performance check: should complete in reasonable time
    assert!(
        duration.as_millis() < 100,
        "Performance check took too long: {:?}",
        duration
    );
}

#[test]
fn test_save_performance_with_trailer_objects() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    // Create many pages
    let mut page_ids = Vec::new();
    for i in 0..100 {
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
            "Kids" => page_ids.iter().map(|&id| Object::Reference(id)).collect::<Vec<_>>(),
            "Count" => page_ids.len() as i64
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id
    });

    let info_id = doc.add_object(dictionary! {
        "Title" => "Performance Test PDF",
        "PageCount" => page_ids.len() as i64
    });

    // Add many custom entries to trailer
    doc.trailer.set("Root", catalog_id);
    doc.trailer.set("Info", info_id);
    for i in 0..10 {
        doc.trailer.set(format!("Custom{}", i).as_bytes(), Object::Integer(i));
    }

    // Measure save performance
    let options = SaveOptions::builder().use_object_streams(true).build();

    let start = Instant::now();
    let mut output = Vec::new();
    doc.save_with_options(&mut output, options).unwrap();
    let save_duration = start.elapsed();

    println!("Saved {} page PDF in {:?}", page_ids.len(), save_duration);
    println!("Output size: {} bytes", output.len());

    // Should complete quickly even with many objects
    assert!(
        save_duration.as_millis() < 500,
        "Save took too long: {:?}",
        save_duration
    );

    // Verify compression worked
    let content = String::from_utf8_lossy(&output);
    assert!(content.contains("/ObjStm"), "Object streams should be created");
}

#[test]
fn test_encryption_check_performance() {
    // Test the specific encryption dictionary check performance
    let mut doc = Document::with_version("1.5");

    // Create an encryption dictionary
    let encrypt_id = doc.add_object(dictionary! {
        "Filter" => "Standard",
        "V" => 2
    });

    // Set it in trailer
    doc.trailer.set("Encrypt", encrypt_id);

    // Create many other objects
    let mut other_ids = Vec::new();
    for i in 0..1000 {
        let id = doc.add_object(Object::Integer(i));
        other_ids.push(id);
    }

    // Time the encryption check
    let start = Instant::now();

    // Check encryption dictionary
    let encrypt_obj = doc.objects.get(&encrypt_id).unwrap();
    let encrypt_compressible = ObjectStream::can_be_compressed(encrypt_id, encrypt_obj, &doc);

    // Check many non-encryption objects
    for &id in &other_ids[..100] {
        // Check first 100
        if let Some(obj) = doc.objects.get(&id) {
            let _ = ObjectStream::can_be_compressed(id, obj, &doc);
        }
    }

    let duration = start.elapsed();

    println!("Encryption check performance: {:?} for 101 objects", duration);

    assert!(!encrypt_compressible, "Encryption dict should not be compressible");
    assert!(duration.as_micros() < 1000, "Check should be very fast");
}
