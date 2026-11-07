use lopdf::content::{Content, Operation};
use lopdf::{Document, Object, Stream, StringFormat, dictionary};

// These ToUnicode CMaps are extracted from a real PDF which I have that have these
// `0 beginbfrange ... endbfrange` sections. Historically (i.e version 0.38.0) this could trigger parse
// failures and break text extraction.
const FONT1_TOUNICODE: &str = r#"/CIDInit /ProcSet findresource begin
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
0 beginbfrange
endbfrange
2 beginbfchar
<0003> <0020>
<0006> <0023>
endbfchar
3 beginbfrange
<0008> <0009> <0025>
<000b> <000c> <0028>
<000f> <0016> <002c>
endbfrange
3 beginbfchar
<001a> <0037>
<001d> <003a>
<0022> <003f>
endbfchar
3 beginbfrange
<0024> <002a> <0041>
<002c> <002d> <0049>
<002f> <0031> <004c>
endbfrange
1 beginbfchar
<0033> <0050>
endbfchar
1 beginbfrange
<0035> <0038> <0052>
endbfrange
2 beginbfchar
<003a> <0057>
<003c> <0059>
endbfchar
2 beginbfrange
<0044> <004c> <0061>
<004e> <005d> <006b>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

const FONT2_TOUNICODE: &str = r#"/CIDInit /ProcSet findresource begin
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
0 beginbfrange
endbfrange
2 beginbfchar
<0003> <0020>
<0007> <0024>
endbfchar
2 beginbfrange
<000a> <000c> <0027>
<000e> <0019> <002b>
endbfrange
1 beginbfchar
<001b> <0038>
endbfchar
5 beginbfrange
<001d> <001f> <003a>
<0021> <0022> <003e>
<0024> <002c> <0041>
<002e> <003b> <004b>
<0044> <005d> <0061>
endbfrange
2 beginbfchar
<0061> <007e>
<00bc> <20ac>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"#;

fn build_doc_with_tounicode(tounicode: &str, encoded_text: Vec<u8>) -> Document {
    let mut doc = Document::with_version("1.5");

    let pages_id = doc.new_object_id();

    let tounicode_bytes = tounicode.as_bytes().to_vec();
    let tounicode_stream_id = doc.add_object(Stream::new(
        dictionary! { "Length" => tounicode_bytes.len() as i64 },
        tounicode_bytes,
    ));

    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type0",
        "BaseFont" => "DummyFont",
        "Encoding" => "Identity-H",
        "ToUnicode" => Object::Reference(tounicode_stream_id),
    });

    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });

    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
            Operation::new("Td", vec![50.into(), 700.into()]),
            Operation::new("Tj", vec![Object::String(encoded_text, StringFormat::Hexadecimal)]),
            Operation::new("ET", vec![]),
        ],
    };

    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

    let page_id = doc.add_object(dictionary! {
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
            "Kids" => vec![page_id.into()],
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

#[test]
fn extract_text_does_not_error_with_empty_bfrange_font2() {
    // Sequence of 2-byte character codes (Identity-H):
    // <0003> -> U+0020 (space)
    // <0007> -> U+0024 ($)
    // <000a> -> U+0027 (') via bfrange
    // <001b> -> U+0038 (8) via bfchar
    // <0061> -> U+007E (~) via bfchar
    // <00bc> -> U+20AC (€) via bfchar
    let encoded_text = vec![
        0x00, 0x03, // space
        0x00, 0x07, // $
        0x00, 0x0a, // '
        0x00, 0x1b, // 8
        0x00, 0x61, // ~
        0x00, 0xbc, // €
    ];

    let doc = build_doc_with_tounicode(FONT2_TOUNICODE, encoded_text);

    let text = doc.extract_text(&[1]).expect("extract_text should not error");
    assert_eq!(text.trim_end(), " $'8~€");
}

#[test]
fn extract_text_does_not_error_with_empty_bfrange_font1() {
    // Exercise both bfchar and bfrange sections, and include the initial empty bfrange section.
    // <0003> -> space
    // <0006> -> #
    // <0008> -> % via bfrange (<0008> <0009> <0025>)
    // <001a> -> 7 via bfchar
    // <0044> -> a via bfrange (<0044> <004c> <0061>)
    let encoded_text = vec![
        0x00, 0x03, // space
        0x00, 0x06, // #
        0x00, 0x08, // %
        0x00, 0x1a, // 7
        0x00, 0x44, // a
    ];

    let doc = build_doc_with_tounicode(FONT1_TOUNICODE, encoded_text);

    let text = doc.extract_text(&[1]).expect("extract_text should not error");
    assert_eq!(text.trim_end(), " #%7a");
}
