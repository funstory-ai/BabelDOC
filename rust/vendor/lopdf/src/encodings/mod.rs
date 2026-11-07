pub mod cmap;
mod differences;
mod glyphnames;
mod mappings;

pub use self::differences::Differences;
pub use self::glyphnames::Glyph;
pub use self::mappings::*;
use crate::Error;
use crate::Result;
use crate::parser_aux::substr;
use cmap::ToUnicodeCMap;
use encoding_rs::UTF_16BE;
use log::debug;

pub fn bytes_to_string(encoding: &CodedCharacterSet, bytes: &[u8], out: &mut String) -> Result<()> {
    for b in bytes {
        let Some(g) = encoding.get(*b as usize).copied().flatten() else {
            continue;
        };

        for ch in char::decode_utf16([g.utf16_code_unit()]).flatten() {
            out.push(ch);
        }
    }

    Ok(())
}

pub fn string_to_bytes(encoding: &CodedCharacterSet, text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    write_to_bytes(encoding, text, &mut out);
    out
}

pub fn write_to_bytes(encoding: &CodedCharacterSet, text: &str, out: &mut Vec<u8>) {
    for c in text.encode_utf16() {
        let g = Glyph::from_utf16_code_unit(c);

        let Some(n) = encoding.iter().position(|glyph| glyph.is_some_and(|f| f == g)) else {
            continue;
        };

        out.push(n as u8);
    }
}

pub enum Encoding<'a> {
    OneByteEncoding(&'a CodedCharacterSet),
    SimpleEncoding(&'a [u8]),
    UnicodeMapEncoding(ToUnicodeCMap),
    Differences(Differences<'a>),
}

impl std::fmt::Debug for Encoding<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // UnicodeCMap and Bytes encoding ommitted to not bloat debug log
            Self::OneByteEncoding(_arg0) => f.debug_tuple("OneByteEncoding").finish(),
            Self::SimpleEncoding(arg0) => f.debug_tuple("SimpleEncoding").field(arg0).finish(),
            Self::UnicodeMapEncoding(_arg0) => f.debug_tuple("UnicodeMapEncoding").finish(),
            Self::Differences(_arg0) => f.debug_tuple("Differences").finish(),
        }
    }
}

impl Encoding<'_> {
    pub fn bytes_to_string(&self, bytes: &[u8]) -> Result<String> {
        let mut out = String::new();
        self.write_to_string(bytes, &mut out)?;
        Ok(out)
    }

    pub fn write_to_string(&self, bytes: &[u8], out: &mut String) -> Result<()> {
        match self {
            Self::OneByteEncoding(map) => {
                bytes_to_string(map, bytes, out)?;
                Ok(())
            }
            Self::SimpleEncoding(b"UniGB-UCS2-H") | Self::SimpleEncoding(b"UniGB-UTF16-H") => {
                out.push_str(UTF_16BE.decode(bytes).0.as_ref());
                Ok(())
            }
            Self::UnicodeMapEncoding(unicode_map) => {
                let mut output_bytes = Vec::new();

                // source codes can have a variadic length from 1 to 4 bytes
                let mut bytes_in_considered_code = 0u8;
                let mut considered_source_code = 0u32;
                for byte in bytes {
                    if bytes_in_considered_code == 4 {
                        let mut value = unicode_map.get_or_replacement_char(considered_source_code, 4);
                        considered_source_code = 0;
                        bytes_in_considered_code = 0;
                        output_bytes.append(&mut value);
                    }
                    bytes_in_considered_code += 1;
                    considered_source_code = considered_source_code * 256 + *byte as u32;
                    if let Some(mut value) = unicode_map.get(considered_source_code, bytes_in_considered_code) {
                        considered_source_code = 0;
                        bytes_in_considered_code = 0;
                        output_bytes.append(&mut value);
                    }
                }
                if bytes_in_considered_code > 0 {
                    let mut value =
                        unicode_map.get_or_replacement_char(considered_source_code, bytes_in_considered_code);
                    output_bytes.append(&mut value);
                }
                let utf16_str: Vec<u8> = output_bytes
                    .iter()
                    .flat_map(|it| [(it / 256) as u8, (it % 256) as u8])
                    .collect();

                out.push_str(UTF_16BE.decode(&utf16_str).0.as_ref());
                Ok(())
            }
            Self::SimpleEncoding(b"WinAnsiEncoding") => {
                bytes_to_string(&WIN_ANSI_ENCODING, bytes, out)?;
                Ok(())
            }
            Self::SimpleEncoding(_) => Err(Error::CharacterEncoding),
            Self::Differences(differences) => differences.bytes_to_string(bytes, out),
        }
    }

    pub fn string_to_bytes(&self, text: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.write_to_bytes(text, &mut bytes);
        bytes
    }

    pub fn write_to_bytes(&self, text: &str, out: &mut Vec<u8>) {
        match self {
            Self::OneByteEncoding(map) => write_to_bytes(map, text, out),
            Self::SimpleEncoding(b"UniGB-UCS2-H") | Self::SimpleEncoding(b"UniGB-UTF16-H") => {
                encode_utf16_be(text, out)
            }
            Self::SimpleEncoding(b"WinAnsiEncoding") => write_to_bytes(&WIN_ANSI_ENCODING, text, out),
            Self::UnicodeMapEncoding(unicode_map) => {
                let mut i = 0;
                while i < text.chars().count() {
                    let current_unicode_seq: Vec<u16> = substr(text, i, 1).encode_utf16().collect();

                    if let Some(entries) = unicode_map.get_source_codes_for_unicode(&current_unicode_seq) {
                        if let Some(entry) = entries.first() {
                            // TODO: Add logic to pick the best entry if multiple
                            let mut bytes_for_code = Vec::new();
                            let val = entry.source_code;
                            match entry.code_len {
                                1 => bytes_for_code.push(val as u8),
                                2 => bytes_for_code.extend_from_slice(&(val as u16).to_be_bytes()),
                                3 => {
                                    bytes_for_code.push((val >> 16) as u8);
                                    bytes_for_code.push((val >> 8) as u8);
                                    bytes_for_code.push(val as u8);
                                }
                                4 => bytes_for_code.extend_from_slice(&val.to_be_bytes()),
                                _ => { /* Should not happen */ }
                            }
                            out.extend(bytes_for_code);
                        } else {
                            // No specific entry, handle as unmappable
                            log::warn!(
                                "Unicode sequence {current_unicode_seq:04X?} found in map but no entries, skipping."
                            );
                        }
                    } else {
                        // Character or sequence not found in CMap
                        log::warn!(
                            "Unicode sequence {current_unicode_seq:04X?} not found in ToUnicode CMap, skipping."
                        );
                    }
                    i += 1;
                }
            }
            Self::SimpleEncoding(_) => {
                debug!("Unknown encoding used to encode text {self:?}");
                out.extend_from_slice(text.as_bytes());
            }
            Self::Differences(differences) => {
                differences.string_to_bytes(text, out);
            }
        }
    }
}

/// Encodes the given `str` to UTF-16BE.
/// The recommended way to encode text strings, as it supports all of
/// unicode and all major PDF readers support it.
pub fn encode_utf16_be(text: &str, out: &mut Vec<u8>) {
    // Prepend BOM to the mark string as UTF-16BE encoded.
    let bom_be: [u8; 2] = [0xFE, 0xFF];
    out.extend_from_slice(&bom_be);
    out.extend(text.encode_utf16().flat_map(|b| b.to_be_bytes()));
}

/// Encodes the given `str` to UTF-8. This method of encoding text strings
/// is first specified in PDF2.0 and reader support is still lacking
/// (notably, Adobe Acrobat Reader doesn't support it at the time of writing).
/// Thus, using it is **NOT RECOMMENDED**.
pub fn encode_utf8(text: &str) -> Vec<u8> {
    // Prepend BOM to the mark string as UTF-8 encoded.
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend(text.bytes());
    bytes
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn unicode_with_2byte_code_does_not_convert_single_bytes() {
        let mut cmap = ToUnicodeCMap::new();

        cmap.put(0x0000, 0x0002, 2, cmap::BfRangeTarget::UTF16CodePoint { offset: 0 });
        cmap.put(0x0024, 0x0025, 2, cmap::BfRangeTarget::UTF16CodePoint { offset: 0 });

        let bytes: [u8; 2] = [0x00, 0x24];

        let result = Encoding::UnicodeMapEncoding(cmap).bytes_to_string(&bytes);

        assert_eq!(result.unwrap(), "\u{0024}");
    }

    #[test]
    fn winansi_bytes_to_string() {
        // 0xe9 = é in WinAnsi, 0xfc = ü, 0xdf = ß
        let bytes = [0x41, 0xe9, 0x42, 0xfc, 0xdf]; // AéBüß
        let result = Encoding::SimpleEncoding(b"WinAnsiEncoding")
            .bytes_to_string(&bytes)
            .expect("WinAnsi decode should succeed");
        assert_eq!(result, "AéBüß");
    }

    #[test]
    fn winansi_string_to_bytes() {
        let text = "Sébastien 0,019€ ü ÄÖÜ ß";
        let bytes = Encoding::SimpleEncoding(b"WinAnsiEncoding").string_to_bytes(text);
        // Round-trip: decode the bytes back via the same encoding
        let decoded = Encoding::OneByteEncoding(&WIN_ANSI_ENCODING)
            .bytes_to_string(&bytes)
            .expect("WinAnsi decode should succeed");
        assert_eq!(decoded, text);
    }
}
