use super::{Encoding, Glyph, Result};
use indexmap::IndexMap;

pub struct Differences<'a> {
    pub base: Box<Encoding<'a>>,
    pub map: IndexMap<u8, Glyph>,
    pub inverse: IndexMap<Glyph, u8>,
}

impl Differences<'_> {
    pub(super) fn bytes_to_string(&self, bytes: &[u8], out: &mut String) -> Result<()> {
        for byte in bytes {
            let Some(glyph) = self.map.get(byte) else {
                self.base.write_to_string(&[*byte], out)?;
                continue;
            };

            for c in char::decode_utf16([glyph.utf16_code_unit()]).flatten() {
                out.push_str(c.encode_utf8(&mut [0; 4]));
            }
        }

        Ok(())
    }

    pub(super) fn string_to_bytes(&self, text: &str, out: &mut Vec<u8>) {
        for c in text.chars() {
            let mut any = false;

            for code_unit in c.encode_utf16(&mut [0; 2]) {
                let glyph = Glyph::from_utf16_code_unit(*code_unit);

                if let Some(byte) = self.inverse.get(&glyph) {
                    out.push(*byte);
                    any = true;
                }
            }

            if any {
                continue;
            }

            let mut b = [0; 4];
            let s = c.encode_utf8(&mut b);
            self.base.write_to_bytes(s, out);
        }
    }
}
