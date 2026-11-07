#[cfg(not(feature = "async"))]
use lopdf::{Document, Object};

#[test]
#[cfg(all(test, not(feature = "async")))]
fn test_get_object() {
    use self::Object;
    use lopdf::Dictionary as LoDictionary;
    use lopdf::Stream as LoStream;

    let mut doc = Document::new();
    let id = doc.add_object(Object::string_literal("test"));
    let id2 = doc.add_object(Object::Stream(LoStream::new(
        LoDictionary::new(),
        "stream".as_bytes().to_vec(),
    )));

    println!("{:?}", id);
    println!("{:?}", id2);

    let obj1_exists = doc.get_object(id).is_ok();
    let obj2_exists = doc.get_object(id2).is_ok();

    assert!(obj1_exists);
    assert!(obj2_exists);
}

#[cfg(all(test, not(feature = "async")))]
mod tests_with_parsing {
    use super::*;
    use lopdf::Result;

    fn modify_text() -> Result<bool> {
        let mut doc = Document::load("assets/example.pdf")?;
        doc.version = "1.4".to_string();
        if let Some(Object::Stream(stream)) = doc.objects.get_mut(&(4, 0)) {
            let mut content = stream.decode_content().unwrap();
            content.operations[3].operands[0] = Object::string_literal("Modified text!");
            stream.set_content(content.encode().unwrap());
        }

        // Create temporary folder to store file.
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test_3_modify.pdf");
        doc.save(file_path)?;
        Ok(true)
    }

    #[test]
    fn test_modify() {
        assert!(modify_text().unwrap());
    }

    fn replace_text() -> Result<Document> {
        let mut doc = Document::load("assets/example.pdf")?;
        doc.replace_text(1, "Hello World!", "Modified text!", None)?;

        // Create temporary folder to store file.
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test_4_unicode_replace.pdf");
        doc.save(&file_path)?;

        let doc = Document::load(file_path)?;
        Ok(doc)
    }

    #[test]
    fn test_replace() {
        assert_eq!(replace_text().unwrap().extract_text(&[1]).unwrap(), "Modified text!\n");
    }

    fn replace_unicode_text() -> Result<Document> {
        let mut doc = Document::load("assets/unicode.pdf")?;
        doc.replace_text(1, "😀", "🔧2", Some("🔨"))?;

        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test_4_unicode_replace.pdf");
        doc.save(&file_path)?;

        let doc = Document::load(file_path)?;
        Ok(doc)
    }

    #[test]
    fn test_unicode_replace() {
        let text = replace_unicode_text().unwrap().extract_text(&[1]).unwrap();
        assert_eq!(text, "🔧🔨\n🔧\n🔨\n");
    }

    fn build_doc_with_tj_array(content_bytes: Vec<u8>) -> Document {
        use lopdf::dictionary;

        let mut doc = Document::with_version("1.5");

        let pages_id = doc.new_object_id();

        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });

        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! {
                "F1" => font_id,
            },
        });

        let content_id = doc.add_object(lopdf::Stream::new(dictionary! {}, content_bytes));

        let single_page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Resources" => resources_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });

        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![single_page_id.into()],
                "Count" => 1,
            }),
        );

        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        doc
    }

    fn replace_text_in_tj_array_doc() -> Result<Document> {
        // Content uses the TJ operator with a single-string array containing
        // all 12 characters of "Hello World!".
        let content = b"BT\n/F1 12 Tf\n100 700 Td\n[(Hello World!)] TJ\nET\n".to_vec();
        let mut doc = build_doc_with_tj_array(content);
        doc.replace_text(1, "Hello World!", "Modified text!", None)?;
        Ok(doc)
    }

    #[test]
    fn test_replace_text_in_tj_array_does_not_truncate() {
        let doc = replace_text_in_tj_array_doc().unwrap();
        let text = doc.extract_text(&[1]).unwrap();
        assert_eq!(text, "Modified text! \n");
    }

    fn get_mut() -> Result<bool> {
        let mut doc = Document::load("assets/example.pdf")?;
        let arr = doc
            .get_object_mut((5, 0))?
            .as_dict_mut()?
            .get_mut(b"Contents")?
            .as_array_mut()?;
        arr[0] = arr[0].clone();
        Ok(true)
    }

    #[test]
    fn test_get_mut() {
        assert!(get_mut().unwrap());
    }
}
