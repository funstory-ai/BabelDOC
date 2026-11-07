use lopdf::content::{Content, Operation};
use lopdf::{Document, Object, Stream, StringFormat, dictionary};

#[test]
fn unicode_can_be_written_to_pdf_and_read() {
    let mut doc = Document::new();

    // literal corresponds to the chars:
    // U+1F600
    // U+1F527
    // U+1F528
    // which are encoded in font cmap with the following hex
    // <005F>
    // <0060>
    // <0061>
    // please mind that indicated BaseFont might not contain
    // those unicode emojis in practice
    let unicode_literal = "😀🔧🔨";
    let literal_encoded_with_cmap = [0x00, 0x5F, 0x00, 0x60, 0x00, 0x61];

    // majority of below code correspond to create document example
    // we are just inserting an appropriate unicode text
    // and Type0 font with approriate characters encoding
    let pages_id = doc.new_object_id();

    let cmap_stream_id = doc.add_object(Stream::new(
        dictionary! {
            "Length" => 437
        },
        b"/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
/Ordering (UCS)
/Supplement 0
>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 beginbfrange
<0000> <005E> <0020>
<005F> <0061> [<D83DDE00> <D83DDD27> <D83DDD28>]
endbfrange
1 beginbfchar
<3A51> <D840DC3E>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end"
        .to_vec(),
    ));

    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type0",
        "BaseFont" => "Ryumin-Light",
        "Encoding" => "Identity-H",
        "ToUnicode" => Object::Reference(cmap_stream_id)
    });

    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });

    // Create content with the prepared emoji string encoded according to values in our font cmap
    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 48.into()]),
            Operation::new("Td", vec![100.into(), 600.into()]),
            Operation::new(
                "Tj",
                vec![Object::String(
                    literal_encoded_with_cmap.to_vec(),
                    StringFormat::Hexadecimal,
                )],
            ),
            Operation::new("ET", vec![]),
        ],
    };

    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
    });

    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
    };

    // using insert() here, instead of add_object() since the id is already known.
    doc.objects.insert(pages_id, Object::Dictionary(pages));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });

    doc.trailer.set("Root", catalog_id);
    doc.compress();

    let extracted_text = get_text_from_first_page(&doc);

    assert_eq!(extracted_text.trim_end(), unicode_literal);
}

fn get_text_from_first_page(doc: &Document) -> String {
    let mut pages = doc.get_pages();
    let first_page = pages.first_entry().expect("Expected pages to be non empty");

    doc.extract_text(&[*first_page.key()])
        .expect("Expected to find text on the first page")
}

#[cfg(not(feature = "async"))]
#[test]
fn unicode_can_be_extracted_from_loaded_pdf() -> lopdf::Result<()> {
    let doc = Document::load("assets/unicode.pdf")?;
    let extracted_text = get_text_from_first_page(&doc);
    // extract text can currently map a consecutive fragment of text
    // to one divided into multiple lines, therefore we have to remove the
    // new lines
    assert_eq!(extracted_text.replace("\n", ""), "😀🔧🔨");
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn unicode_can_be_extracted_from_loaded_pdf() -> lopdf::Result<()> {
    let doc = Document::load("assets/unicode.pdf").await?;
    let extracted_text = get_text_from_first_page(&doc);
    assert_eq!(extracted_text.replace("\n", ""), "😀🔧🔨");
    Ok(())
}
