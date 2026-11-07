//! Saving an encrypted document with object streams enabled.
//!
//! Object streams are assembled while serializing, long after `Document::encrypt`
//! has encrypted every object and dropped the file encryption key, so the writer
//! has nothing left to encrypt one with.
//! See https://github.com/J-F-Liu/lopdf/issues/479.

#![cfg(not(feature = "async"))]

use lopdf::{
    Document, EncryptionState, EncryptionVersion, LoadOptions, Object, Permissions, SaveOptions, Stream, StringFormat,
    dictionary,
};

/// Build a one page document carrying a string we can check after a round trip.
fn sample_document() -> Document {
    let mut doc = Document::with_version("1.7");

    let pages_id = doc.new_object_id();
    let content_id = doc.add_object(Stream::new(dictionary! {}, b"BT ET".to_vec()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => Object::Reference(content_id),
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![Object::Reference(page_id)],
            "Count" => 1,
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id),
    });
    let info_id = doc.add_object(dictionary! {
        "Title" => Object::String(b"lopdf 479".to_vec(), StringFormat::Literal),
    });

    doc.trailer.set("Root", Object::Reference(catalog_id));
    doc.trailer.set("Info", Object::Reference(info_id));
    let id = vec![0x42u8; 16];
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(id.clone(), StringFormat::Literal),
            Object::String(id, StringFormat::Literal),
        ]),
    );

    doc
}

/// AES-128, the setup from the issue report.
fn encrypt_aes128(doc: &mut Document) {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    let crypt_filter: Arc<dyn lopdf::encryption::crypt_filters::CryptFilter> =
        Arc::new(lopdf::encryption::crypt_filters::Aes128CryptFilter);
    let version = EncryptionVersion::V4 {
        document: doc,
        encrypt_metadata: true,
        crypt_filters: BTreeMap::from([(b"StdCF".to_vec(), crypt_filter)]),
        stream_filter: b"StdCF".to_vec(),
        string_filter: b"StdCF".to_vec(),
        owner_password: "owner",
        user_password: "",
        permissions: Permissions::PRINTABLE,
    };
    let state = EncryptionState::try_from(version).unwrap();
    doc.encrypt(&state).unwrap();
}

/// RC4-128, to show the problem is not specific to the AES crypt filters.
fn encrypt_rc4_128(doc: &mut Document) {
    let version = EncryptionVersion::V2 {
        document: doc,
        owner_password: "owner",
        user_password: "",
        key_length: 128,
        permissions: Permissions::PRINTABLE,
    };
    let state = EncryptionState::try_from(version).unwrap();
    doc.encrypt(&state).unwrap();
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|window| window == needle)
}

/// Assert the saved bytes are a readable encrypted PDF.
fn assert_round_trips(buffer: &[u8], label: &str) {
    let doc = Document::load_mem_with_options(buffer, LoadOptions::with_password("owner"))
        .unwrap_or_else(|err| panic!("{label}: reloading the saved document failed: {err:?}"));

    assert!(doc.was_encrypted(), "{label}: document should have been encrypted");

    let pages = doc.get_pages();
    assert_eq!(pages.len(), 1, "{label}: expected exactly one page");

    // Before the fix this came back as the raw ciphertext of the title: the object
    // holding it was packed into an object stream, so the reader decrypts the stream
    // as a whole and correctly leaves the strings inside it alone, but `encrypt` had
    // already encrypted them individually.
    let title = doc
        .trailer
        .get(b"Info")
        .and_then(Object::as_reference)
        .and_then(|id| doc.get_dictionary(id))
        .and_then(|dict| dict.get(b"Title"))
        .and_then(Object::as_str)
        .unwrap_or_else(|err| panic!("{label}: /Title is unreadable: {err:?}"));
    assert_eq!(title, b"lopdf 479", "{label}: /Title did not survive the round trip");

    let page_id = *pages.get(&1).unwrap();
    let content = doc.get_page_content(page_id);
    assert_eq!(
        content.trim_ascii_end(),
        b"BT ET",
        "{label}: page content did not survive the round trip"
    );
}

#[test]
fn save_modern_keeps_encrypted_documents_readable() {
    let mut doc = sample_document();
    encrypt_aes128(&mut doc);

    let mut buffer = Vec::new();
    doc.save_modern(&mut buffer).unwrap();

    assert_round_trips(&buffer, "save_modern");
}

#[test]
fn save_with_object_streams_keeps_encrypted_rc4_documents_readable() {
    let mut doc = sample_document();
    encrypt_rc4_128(&mut doc);

    let options = SaveOptions::builder()
        .use_object_streams(true)
        .use_xref_streams(false)
        .build();
    let mut buffer = Vec::new();
    doc.save_with_options(&mut buffer, options).unwrap();

    assert_round_trips(&buffer, "save_with_options");
}

#[test]
fn save_modern_writes_no_object_stream_when_encrypted() {
    let mut doc = sample_document();
    encrypt_aes128(&mut doc);

    let mut buffer = Vec::new();
    doc.save_modern(&mut buffer).unwrap();

    assert!(
        !contains(&buffer, b"/ObjStm"),
        "an encrypted document must not be packed into an object stream the writer cannot encrypt"
    );
    // The fallback drops object streams only. Cross-reference streams are never
    // encrypted, so `save_modern` still gets the modern cross-reference section.
    assert!(
        contains(&buffer, b"/XRef"),
        "the requested cross-reference stream should still be written"
    );
}

#[test]
fn save_modern_still_uses_object_streams_when_not_encrypted() {
    let mut doc = sample_document();

    let mut buffer = Vec::new();
    doc.save_modern(&mut buffer).unwrap();

    assert!(
        contains(&buffer, b"/ObjStm"),
        "an unencrypted document should still be packed into object streams"
    );
}
