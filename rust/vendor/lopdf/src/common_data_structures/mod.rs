use crate::{
    Error, Object, Result, StringFormat,
    encodings::{self, bytes_to_string},
};

/// Creates a text string.
/// If the input only contains ASCII characters, the string is encoded
/// in PDFDocEncoding, otherwise in UTF-16BE.
pub fn text_string(text: &str) -> Object {
    if text.is_ascii() {
        return Object::String(text.into(), StringFormat::Literal);
    }
    let mut string = Vec::new();
    encodings::encode_utf16_be(text, &mut string);
    Object::String(string, StringFormat::Hexadecimal)
}

/// Decodes a text string.
/// Depending on the BOM at the start of the string, a different encoding is chosen.
/// All encodings specified in PDF2.0 are supported (PDFDocEncoding, UTF-16BE,
/// and UTF-8).
pub fn decode_text_string(obj: &Object) -> Result<String> {
    let s = obj.as_str()?;
    if s.starts_with(b"\xFE\xFF") {
        // Detected UTF-16BE BOM
        String::from_utf16(
            &s[2..]
                .chunks(2)
                .map(|c| {
                    if c.len() == 1 {
                        u16::from_be_bytes([c[0], 0])
                    } else {
                        u16::from_be_bytes(c.try_into().unwrap())
                    }
                })
                .collect::<Vec<u16>>(),
        )
        .map_err(|_| Error::TextStringDecode)
    } else if s.starts_with(b"\xEF\xBB\xBF") {
        // Detected UTF-8 BOM
        String::from_utf8(s.to_vec()).map_err(|_| Error::TextStringDecode)
    } else {
        // If neither BOM is detected, PDFDocEncoding is used
        let mut out = String::new();
        bytes_to_string(&encodings::PDF_DOC_ENCODING, s, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        Object, StringFormat, common_data_structures::decode_text_string, encodings, text_string, writer::Writer,
    };

    #[test]
    fn spec_example1_encode() {
        let input = "text‰";
        let text_string = encodings::string_to_bytes(&encodings::PDF_DOC_ENCODING, input);
        // let text_string = input.bytes().collect::<Vec<_>>();
        let dict = Object::Dictionary(dictionary!(
            "Key" => Object::String(text_string, StringFormat::Literal),
        ));
        let mut actual = vec![];
        Writer::write_object(&mut actual, &dict).unwrap();
        // "\x8B" is equivalent to the escaped version "\\213" which is used
        // in the original example.
        let expected = b"<</Key(text\x8B)>>";
        assert_eq!(actual.as_slice(), expected);
    }

    #[test]
    fn spec_example1_decode() {
        let input = b"<</Key(text\\213)>>";
        let dict = crate::parser::direct_object(input).unwrap();
        let dict = dict.as_dict().unwrap();
        let actual = decode_text_string(dict.get(b"Key").unwrap()).unwrap();
        let expected = "text‰";
        assert_eq!(&actual, expected);
    }

    #[test]
    fn spec_example2_encode() {
        // Russian for "test"
        let input = "тест";
        // let text_string = encodings::string_to_bytes(encodings::PDF_DOC_ENCODING, input);
        let dict = Object::Dictionary(dictionary!(
            "Key" => text_string(input),
        ));
        let mut actual = vec![];
        Writer::write_object(&mut actual, &dict).unwrap();
        let expected = b"<</Key<FEFF0442043504410442>>>";
        assert_eq!(actual.as_slice(), expected);
    }

    #[test]
    fn spec_example2_decode() {
        let input = b"<</Key<FEFF0442043504410442>>>";
        let dict = crate::parser::direct_object(input).unwrap();
        let dict = dict.as_dict().unwrap();
        let actual = decode_text_string(dict.get(b"Key").unwrap()).unwrap();
        // Russian for "test"
        let expected = "тест";
        assert_eq!(&actual, expected);
    }
}
