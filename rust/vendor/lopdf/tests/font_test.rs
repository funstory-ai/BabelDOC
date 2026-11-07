#[cfg(feature = "font_embedding")]
#[test]
fn test_font_data_creation() {
    // Load a sample TTF font file from the test resources
    let font_file = std::fs::read("./assets/fonts/Montserrat-Regular.ttf");

    // Ensure the font file was read successfully
    assert!(font_file.is_ok(), "Failed to read font file");

    // Unwrap the result to get the font file bytes
    let font_file = font_file.unwrap();
    let mut font_data = lopdf::FontData::new(&font_file, "Montserrat-Regular".to_string());

    // Create a new FontData instance
    font_data
        .set_font_bbox((0, -200, 1000, 800))
        .set_italic_angle(10)
        .set_ascent(800)
        .set_descent(-200)
        .set_cap_height(700)
        .set_stem_v(100)
        .set_flags(4)
        .set_encoding("WinAnsiEncoding".to_string());

    // Verify the properties of the FontData instance
    assert_eq!(font_data.font_bbox, (0, -200, 1000, 800));
    assert_eq!(font_data.italic_angle, 10);
    assert_eq!(font_data.ascent, 800);
    assert_eq!(font_data.descent, -200);
    assert_eq!(font_data.cap_height, 700);
    assert_eq!(font_data.stem_v, 100);
    assert_eq!(font_data.flags, 4);
    assert_eq!(font_data.encoding, "WinAnsiEncoding");
    assert!(!font_data.bytes().is_empty(), "Font data should not be empty");
}
