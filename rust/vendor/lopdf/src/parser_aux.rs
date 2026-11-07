use log::warn;

use crate::{Dictionary, Object, ObjectId, Stream, parser};
use crate::{
    Error, Result,
    content::{Content, Operation},
    document::Document,
    encodings::Encoding,
    error::ParseError,
    object::Object::Name,
    xref::{Xref, XrefEntry, XrefType},
};
use std::{
    collections::BTreeMap,
    io::{Cursor, Read},
};

impl Content<Vec<Operation>> {
    /// Decode content operations.
    pub fn decode(data: &[u8]) -> Result<Self> {
        parser::content(data).ok_or(ParseError::InvalidContentStream.into())
    }

    /// Strict decode content operations.
    pub fn decode_strict(data: &[u8]) -> Result<Self> {
        parser::content_strict(data).map_err(|e| e.into())
    }
}

impl Stream {
    /// Decode content after decoding all stream filters.
    pub fn decode_content(&self) -> Result<Content<Vec<Operation>>> {
        Content::decode(&self.content)
    }
}

impl Document {
    /// Get decoded page content;
    pub fn get_and_decode_page_content(&self, page_id: ObjectId) -> Result<Content<Vec<Operation>>> {
        let content_data = self.get_page_content(page_id);
        Content::decode(&content_data)
    }

    /// Add content to a page. All existing content will be unchanged.
    pub fn add_to_page_content(&mut self, page_id: ObjectId, content: Content<Vec<Operation>>) -> Result<()> {
        let content_data = Content::encode(&content)?;
        self.add_page_contents(page_id, content_data)?;
        Ok(())
    }

    pub fn extract_text(&self, page_numbers: &[u32]) -> Result<String> {
        self.extract_text_inner(page_numbers, None)
    }

    /// Extract text from the given pages, bounding the total decompressed content
    /// of each page to `max_decompressed_size` bytes.
    ///
    /// This is the decompression-bomb-safe counterpart to
    /// [`Document::extract_text`]: page content is decoded via
    /// [`Document::get_page_content_with_limit`], so a small compressed content
    /// stream cannot inflate without limit. Prefer it over [`Document::extract_text`]
    /// for PDFs from an untrusted source. Returns
    /// [`DecompressError::MemoryLimitExceeded`](crate::DecompressError::MemoryLimitExceeded)
    /// if any requested page's content would exceed the limit.
    pub fn extract_text_with_limit(&self, page_numbers: &[u32], max_decompressed_size: usize) -> Result<String> {
        self.extract_text_inner(page_numbers, Some(max_decompressed_size))
    }

    fn extract_text_inner(&self, page_numbers: &[u32], limit: Option<usize>) -> Result<String> {
        let text_fragments = self.extract_text_chunks_inner(page_numbers, limit);
        let mut text = String::new();
        for maybe_text_fragment in text_fragments.into_iter() {
            let text_fragment = maybe_text_fragment?;
            text.push_str(&text_fragment);
        }

        Ok(text)
    }

    pub fn extract_text_chunks(&self, page_numbers: &[u32]) -> Vec<Result<String>> {
        self.extract_text_chunks_inner(page_numbers, None)
    }

    /// Bomb-safe counterpart to [`Document::extract_text_chunks`], bounding the
    /// total decompressed content of each page to `max_decompressed_size` bytes.
    /// A page whose content exceeds the limit yields an `Err` chunk carrying
    /// [`DecompressError::MemoryLimitExceeded`](crate::DecompressError::MemoryLimitExceeded).
    pub fn extract_text_chunks_with_limit(
        &self, page_numbers: &[u32], max_decompressed_size: usize,
    ) -> Vec<Result<String>> {
        self.extract_text_chunks_inner(page_numbers, Some(max_decompressed_size))
    }

    fn extract_text_chunks_inner(&self, page_numbers: &[u32], limit: Option<usize>) -> Vec<Result<String>> {
        let pages: BTreeMap<u32, (u32, u16)> = self.get_pages();
        page_numbers
            .iter()
            .flat_map(|page_number| {
                let result = self.extract_text_chunks_from_page(&pages, *page_number, limit);
                match result {
                    Ok(text_chunks) => text_chunks,
                    Err(err) => vec![Err(err)],
                }
            })
            .collect()
    }

    fn extract_text_chunks_from_page(
        &self, pages: &BTreeMap<u32, (u32, u16)>, page_number: u32, limit: Option<usize>,
    ) -> Result<Vec<Result<String>>> {
        let mut collected_chunks_and_errs: Vec<std::result::Result<String, Error>> = Vec::new();

        let page_id = *pages.get(&page_number).ok_or(Error::PageNumberNotFound(page_number))?;
        let fonts = self.get_page_fonts(page_id)?;
        let encodings: BTreeMap<Vec<u8>, Encoding> = fonts
            .into_iter()
            .filter_map(|(name, font)| {
                let encoding = match limit {
                    Some(max) => font.get_font_encoding_with_limit(self, max),
                    None => font.get_font_encoding(self),
                };
                match encoding {
                    Ok(it) => Some((name, it)),
                    Err(err) => {
                        collected_chunks_and_errs.push(Err(err));
                        None
                    }
                }
            })
            .collect();
        let content_data = match limit {
            Some(max) => self.get_page_content_with_limit(page_id, max)?,
            None => self.get_page_content(page_id),
        };
        let content = Content::decode(&content_data)?;

        // each text with different encoding is extracted as separate chunk
        let mut current_encoding = None;
        let mut current_text = String::new();
        for operation in &content.operations {
            match operation.operator.as_ref() {
                "Tf" => {
                    let current_font = operation
                        .operands
                        .first()
                        .ok_or_else(|| Error::Syntax("missing font operand".to_string()))?
                        .as_name();
                    current_encoding = match current_font {
                        Ok(font) => encodings.get(font),
                        Err(err) => {
                            collected_chunks_and_errs.push(Err(err));
                            None
                        }
                    };

                    if !current_text.is_empty() {
                        collected_chunks_and_errs.push(Ok(current_text));
                        current_text = String::new();
                    }
                }
                "Tj" | "TJ" => match current_encoding {
                    Some(encoding) => {
                        let res = collect_text(&mut current_text, encoding, &operation.operands);
                        if let Err(err) = res {
                            collected_chunks_and_errs.push(Err(err));
                        }
                    }
                    None => warn!("Could not decode extracted text"),
                },
                // PDF 32000-1 §9.4.3 — `'` is equivalent to `T* Tj`:
                // move to next line, then show string from the single
                // string operand.
                "'" => match current_encoding {
                    Some(encoding) => {
                        if !current_text.ends_with('\n') {
                            current_text.push('\n');
                        }
                        let res = collect_text(&mut current_text, encoding, &operation.operands);
                        if let Err(err) = res {
                            collected_chunks_and_errs.push(Err(err));
                        }
                    }
                    None => warn!("Could not decode extracted text"),
                },
                // PDF 32000-1 §9.4.3 — `"` is equivalent to
                // `aw Tw ac Tc T* Tj` with operands `[aw, ac, string]`.
                // Operands 0/1 set word/character spacing for rendering
                // and don't affect the extracted character sequence;
                // operand 2 is the string to show.
                "\"" => match current_encoding {
                    Some(encoding) => {
                        if !current_text.ends_with('\n') {
                            current_text.push('\n');
                        }
                        if let Some(string_operand) = operation.operands.get(2) {
                            let res = collect_text(&mut current_text, encoding, std::slice::from_ref(string_operand));
                            if let Err(err) = res {
                                collected_chunks_and_errs.push(Err(err));
                            }
                        }
                    }
                    None => warn!("Could not decode extracted text"),
                },
                // PDF 32000-1 §9.4.2 — `T*` moves to the start of the
                // next line. For text extraction we approximate this
                // as `\n`, matching how the `ET` arm above handles end
                // of text object.
                "T*" if !current_text.ends_with('\n') => current_text.push('\n'),
                "T*" => {}
                "ET" if !current_text.ends_with('\n') => current_text.push('\n'),
                "ET" => {}
                _ => {}
            }
        }
        if !current_text.is_empty() {
            collected_chunks_and_errs.push(Ok(current_text));
        }

        Ok(collected_chunks_and_errs)
    }

    pub fn replace_text(
        &mut self, page_number: u32, text: &str, other_text: &str, default_str: Option<&str>,
    ) -> Result<()> {
        let page = page_number.saturating_sub(1) as usize;
        let page_id = self
            .page_iter()
            .nth(page)
            .ok_or(Error::PageNumberNotFound(page_number))?;
        let encodings: BTreeMap<Vec<u8>, Encoding> = self
            .get_page_fonts(page_id)?
            .into_iter()
            .map(|(name, font)| font.get_font_encoding(self).map(|it| (name, it)))
            .collect::<Result<BTreeMap<Vec<u8>, Encoding>>>()?;
        let content_data = self.get_page_content(page_id);
        let mut content = Content::decode(&content_data)?;
        let mut current_encoding = None;
        for operation in &mut content.operations {
            match operation.operator.as_ref() {
                "Tf" => {
                    let current_font = operation
                        .operands
                        .first()
                        .ok_or_else(|| Error::Syntax("missing font operand".to_string()))?
                        .as_name()?;
                    current_encoding = encodings.get(current_font);
                }
                "Tj" | "TJ" => match current_encoding {
                    Some(encoding) => {
                        try_to_replace_encoded_text(operation, encoding, text, other_text, default_str.unwrap_or(""))?
                    }
                    None => {
                        warn!("Could not decode extracted text, some of the occurances might not be properly replaced")
                    }
                },
                _ => {}
            }
        }
        let modified_content = content.encode()?;
        self.change_page_content(page_id, modified_content)
    }

    pub fn replace_partial_text(
        &mut self, page_number: u32, search_text: &str, replacement_text: &str, default_char: Option<&str>,
    ) -> Result<usize> {
        let page = page_number.saturating_sub(1) as usize;
        let page_id = self
            .page_iter()
            .nth(page)
            .ok_or(Error::PageNumberNotFound(page_number))?;

        let encodings: BTreeMap<Vec<u8>, Encoding> = self
            .get_page_fonts(page_id)?
            .into_iter()
            .map(|(name, font)| font.get_font_encoding(self).map(|it| (name, it)))
            .collect::<Result<BTreeMap<Vec<u8>, Encoding>>>()?;

        let content_data = self.get_page_content(page_id);
        let mut content = Content::decode(&content_data)?;
        let mut current_encoding = None;
        let mut replacement_count = 0;

        for operation in &mut content.operations {
            match operation.operator.as_ref() {
                "Tf" => {
                    let current_font = operation
                        .operands
                        .first()
                        .ok_or_else(|| Error::Syntax("missing font operand".to_string()))?
                        .as_name()?;
                    current_encoding = encodings.get(current_font);
                }
                "Tj" | "TJ" => {
                    if let Some(encoding) = current_encoding {
                        replacement_count += replace_partial_in_operation(
                            operation,
                            encoding,
                            search_text,
                            replacement_text,
                            default_char.unwrap_or("?"),
                        )?;
                    } else {
                        warn!("No encoding found for text operation");
                    }
                }
                _ => {}
            }
        }

        if replacement_count > 0 {
            let modified_content = content.encode()?;
            self.change_page_content(page_id, modified_content)?;
        }

        Ok(replacement_count)
    }

    pub fn insert_image(
        &mut self, page_id: ObjectId, img_object: Stream, position: (f32, f32), size: (f32, f32),
    ) -> Result<()> {
        let img_id = self.add_object(img_object);
        let img_name = format!("X{}", img_id.0);

        self.add_xobject(page_id, img_name.as_bytes(), img_id)?;

        let mut content = self.get_and_decode_page_content(page_id)?;
        content.operations.push(Operation::new("q", vec![]));
        content.operations.push(Operation::new(
            "cm",
            vec![
                size.0.into(),
                0.into(),
                0.into(),
                size.1.into(),
                position.0.into(),
                position.1.into(),
            ],
        ));
        content
            .operations
            .push(Operation::new("Do", vec![Name(img_name.as_bytes().to_vec())]));
        content.operations.push(Operation::new("Q", vec![]));

        self.change_page_content(page_id, content.encode()?)
    }

    pub fn insert_form_object(&mut self, page_id: ObjectId, form_obj: Stream) -> Result<()> {
        let form_id = self.add_object(form_obj);
        let form_name = format!("X{}", form_id.0);

        let mut content = self.get_and_decode_page_content(page_id)?;
        content.operations.insert(0, Operation::new("q", vec![]));
        content.operations.push(Operation::new("Q", vec![]));
        content
            .operations
            .push(Operation::new("Do", vec![Name(form_name.as_bytes().to_vec())]));
        let modified_content = content.encode()?;
        self.add_xobject(page_id, form_name, form_id)?;

        self.change_page_content(page_id, modified_content)
    }
}
fn collect_text(text: &mut String, encoding: &Encoding, operands: &[Object]) -> Result<()> {
    for operand in operands.iter() {
        match operand {
            Object::String(bytes, _) => {
                encoding.write_to_string(bytes, text)?;
            }
            Object::Array(arr) => {
                collect_text(text, encoding, arr)?;
                text.push(' ');
            }
            Object::Integer(i) if *i < -100 => {
                text.push(' ');
            }
            _ => {}
        }
    }
    Ok(())
}
pub fn substr(s: &str, start: usize, len: usize) -> &str {
    let mut indices = s.char_indices();

    for _ in 0..start {
        if indices.next().is_none() {
            return "";
        }
    }

    let Some((start_idx, _)) = indices.next() else {
        return "";
    };

    let end_idx = indices
        .nth(len.saturating_sub(1))
        .map(|(idx, _)| idx)
        .unwrap_or(s.len());

    &s[start_idx..end_idx]
}
pub fn substring(s: &str, start: usize) -> &str {
    s.char_indices().nth(start).map(|(idx, _)| &s[idx..]).unwrap_or("")
}

fn encode(encoding: &Encoding, txt: &str, default_str: &str) -> Vec<u8> {
    if txt.chars().count() > 1 {
        let mut cur = 0;
        let mut result = Vec::new();
        while cur < txt.chars().count() {
            let c = substr(txt, cur, 1);
            result.extend_from_slice(&encode(encoding, c, default_str));
            cur += 1;
        }
        result
    } else {
        let encoded_bytes = Document::encode_text(encoding, txt);
        if !encoded_bytes.is_empty() {
            encoded_bytes
        } else {
            Document::encode_text(encoding, default_str)
        }
    }
}
fn try_to_replace_encoded_text(
    operation: &mut Operation, encoding: &Encoding, text_to_replace: &str, replacement: &str, default_str: &str,
) -> Result<()> {
    for operand in &mut operation.operands {
        match operand {
            Object::String(bytes, _) => {
                let decoded_text = Document::decode_text(encoding, bytes)?;
                if decoded_text == text_to_replace {
                    let encoded_bytes = encode(encoding, replacement, default_str);
                    *bytes = encoded_bytes;
                }
            }
            Object::Array(arr) => {
                let mut str_collected = String::new();
                collect_text(&mut str_collected, encoding, arr)?;
                if str_collected == text_to_replace {
                    // The number of `Object::String` items in a `TJ` array is
                    // not guaranteed to match the character count of the
                    // decoded text (each string may hold several glyphs, and
                    // numeric kerning entries are interleaved).
                    //
                    // There is no **good** way to interpolate between the OG
                    // and the replacement, but putting the full encoded replacement
                    // into the first string slot and emptying out the remaining
                    // string slots, leaving any numeric kerning entries in place
                    // seems like the least bad option.
                    let encoded_replacement = encode(encoding, replacement, default_str);
                    let mut placed = false;
                    for item in arr.iter_mut() {
                        if let Object::String(bytes, _f) = item {
                            if placed {
                                *bytes = Vec::new();
                            } else {
                                bytes.clone_from(&encoded_replacement);
                                placed = true;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn replace_partial_in_operation(
    operation: &mut Operation, encoding: &Encoding, search_text: &str, replacement_text: &str, default_char: &str,
) -> Result<usize> {
    let mut replacement_count = 0;

    for operand in &mut operation.operands {
        match operand {
            Object::String(bytes, _) => {
                let decoded_text = Document::decode_text(encoding, bytes)?;
                if decoded_text.contains(search_text) {
                    let new_text = decoded_text.replace(search_text, replacement_text);
                    let encoded_bytes = encode_with_fallback(encoding, &new_text, default_char);
                    *bytes = encoded_bytes;
                    replacement_count += decoded_text.matches(search_text).count();
                }
            }
            Object::Array(arr) => {
                replacement_count +=
                    replace_partial_in_array(arr, encoding, search_text, replacement_text, default_char)?;
            }
            _ => {}
        }
    }

    Ok(replacement_count)
}

fn replace_partial_in_array(
    arr: &mut [Object], encoding: &Encoding, search_text: &str, replacement_text: &str, default_char: &str,
) -> Result<usize> {
    let mut replacement_count = 0;

    for item in arr.iter_mut() {
        if let Object::String(bytes, _) = item {
            let decoded_text = Document::decode_text(encoding, bytes)?;
            if decoded_text.contains(search_text) {
                let new_text = decoded_text.replace(search_text, replacement_text);
                let encoded_bytes = encode_with_fallback(encoding, &new_text, default_char);
                *bytes = encoded_bytes;
                replacement_count += decoded_text.matches(search_text).count();
            }
        }
    }

    Ok(replacement_count)
}

fn encode_with_fallback(encoding: &Encoding, text: &str, default_char: &str) -> Vec<u8> {
    let encoded = Document::encode_text(encoding, text);
    if !encoded.is_empty() {
        return encoded;
    }

    encode(encoding, text, default_char)
}

/// Decode CrossReferenceStream
pub fn decode_xref_stream(stream: Stream) -> Result<(Xref, Dictionary)> {
    decode_xref_stream_with_limit(stream, None)
}

/// Decode a cross-reference stream, rejecting it if its decompressed content
/// would exceed `max_decompressed_size` bytes. `None` means no limit (the
/// behavior of [`decode_xref_stream`]). Cross-reference streams are decoded
/// early during loading, so this bounds the memory a `/XRef` stream can use.
pub fn decode_xref_stream_with_limit(
    stream: Stream, max_decompressed_size: Option<usize>,
) -> Result<(Xref, Dictionary)> {
    decode_xref_stream_with_policy(stream, max_decompressed_size, None)
}

pub(crate) fn decode_xref_stream_with_policy(mut stream: Stream, max_decompressed_size: Option<usize>, policy: Option<&crate::resource::Policy>) -> Result<(Xref, Dictionary)> {
    if stream.is_compressed() {
        let decoded = stream.decompressed_content_with_policy(policy, max_decompressed_size)?;
        stream.set_plain_content(decoded);
    }
    let mut dict = stream.dict;
    let mut reader = Cursor::new(stream.content);
    let size = dict
        .get(b"Size")
        .and_then(Object::as_i64)
        .map_err(|_| ParseError::InvalidXref)?;
    let mut xref = Xref::new(size as u32, XrefType::CrossReferenceStream);
    {
        let section_indice = dict
            .get(b"Index")
            .and_then(parse_integer_array)
            .unwrap_or_else(|_| vec![0, size]);
        let field_widths = dict
            .get(b"W")
            .and_then(parse_integer_array)
            .map_err(|_| ParseError::InvalidXref)?;

        // Each width is a byte count for one xref-stream field. The PDF spec doesn't need more than a
        // few bytes per field (8 covers the full u64/i64 range); bound them before sizing a Vec from
        // them, or a crafted stream can request an allocation of up to (2^64-1) bytes per field.
        const MAX_XREF_FIELD_WIDTH: i64 = 8;
        if field_widths.len() < 3
            || field_widths[0].is_negative()
            || field_widths[1].is_negative()
            || field_widths[2].is_negative()
            || field_widths[0] > MAX_XREF_FIELD_WIDTH
            || field_widths[1] > MAX_XREF_FIELD_WIDTH
            || field_widths[2] > MAX_XREF_FIELD_WIDTH
        {
            return Err(ParseError::InvalidXref.into());
        }

        // Total bytes one entry consumes. With every width zero an entry reads nothing, so the loop
        // below would run purely on the attacker-controlled /Index counts and never reach the end of
        // the stream. Such a stream carries no information and can only be malformed, so reject it
        // (this also keeps the body-capacity check below from dividing by zero).
        let entry_width = (field_widths[0] + field_widths[1] + field_widths[2]) as usize;
        if entry_width == 0 {
            return Err(ParseError::InvalidXref.into());
        }

        let index_entries = section_indice
            .as_chunks::<2>()
            .0
            .iter()
            .try_fold(0_usize, |total, section| {
                let count = usize::try_from(section[1]).map_err(|_| ParseError::InvalidXref)?;
                total.checked_add(count).ok_or(ParseError::InvalidXref)
            })?;
        // An entry can't be read from bytes that aren't there. Validate the total before inserting
        // anything so multiple individually plausible /Index sections cannot overrun the body.
        //
        // deliberate: Records narrower than three bytes are rejected to limit amplification from
        // decoded bytes into retained map entries. /W [1 2 0] is a plausible three-byte record and
        // remains accepted; lopdf's writer emits /W [1 4 2], so its output is unaffected.
        const MIN_ENTRY_WIDTH: usize = 3;
        if index_entries > reader.get_ref().len() / entry_width.max(MIN_ENTRY_WIDTH) {
            return Err(ParseError::InvalidXref.into());
        }

        let mut bytes1 = vec![0_u8; field_widths[0] as usize];
        let mut bytes2 = vec![0_u8; field_widths[1] as usize];
        let mut bytes3 = vec![0_u8; field_widths[2] as usize];

        for section in section_indice.as_chunks::<2>().0 {
            let start = section[0];
            let count = section[1];

            for j in 0..count {
                let entry_type = if !bytes1.is_empty() {
                    read_big_endian_integer(&mut reader, bytes1.as_mut_slice())?
                } else {
                    1
                };
                match entry_type {
                    0 => {
                        // free object
                        read_big_endian_integer(&mut reader, bytes2.as_mut_slice())?;
                        read_big_endian_integer(&mut reader, bytes3.as_mut_slice())?;
                    }
                    1 => {
                        // normal object
                        let offset = read_big_endian_integer(&mut reader, bytes2.as_mut_slice())?;
                        let generation = if !bytes3.is_empty() {
                            read_big_endian_integer(&mut reader, bytes3.as_mut_slice())?
                        } else {
                            0
                        } as u16;
                        xref.insert((start + j) as u32, XrefEntry::Normal { offset, generation });
                    }
                    2 => {
                        // compressed object
                        let container = read_big_endian_integer(&mut reader, bytes2.as_mut_slice())?;
                        let index = read_big_endian_integer(&mut reader, bytes3.as_mut_slice())? as u16;
                        xref.insert((start + j) as u32, XrefEntry::Compressed { container, index });
                    }
                    _ => {}
                }
            }
        }
    }
    dict.remove(b"Length");
    dict.remove(b"W");
    dict.remove(b"Index");
    Ok((xref, dict))
}

fn read_big_endian_integer<T: AsRef<[u8]>>(reader: &mut Cursor<T>, buffer: &mut [u8]) -> Result<u32> {
    reader.read_exact(buffer)?;
    let mut value = 0;
    for &mut byte in buffer {
        value = (value << 8) + u32::from(byte);
    }
    Ok(value)
}

fn parse_integer_array(array: &Object) -> Result<Vec<i64>> {
    let array = array.as_array()?;
    let mut out = Vec::with_capacity(array.len());

    for n in array {
        out.push(n.as_i64()?);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "async"))]
    #[test]
    fn load_and_save() {
        use crate::Document;
        use crate::creator::tests::{create_document, save_document};

        // test load_from() and save_to()
        use std::fs::File;
        use std::io::Cursor;
        // Create temporary folder to store file.
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_1_load_and_save.pdf");

        let mut doc = create_document();

        save_document(&file_path, &mut doc);

        let in_file = File::open(file_path).unwrap();
        let mut in_doc = Document::load_from(in_file).unwrap();

        let out_buf = Vec::new();
        let mut memory_cursor = Cursor::new(out_buf);
        in_doc.save_to(&mut memory_cursor).unwrap();
        // Check if saved file is not an empty bytes vector.
        assert!(!memory_cursor.get_ref().is_empty());
    }

    #[test]
    fn extract_text_chunks() {
        use crate::creator::tests::create_document_with_texts;

        let text1 = "Hello world!";
        let text2 = "Ferris is the best!";
        let doc = create_document_with_texts(&[text1, text2]);
        let extracted_texts = doc.extract_text_chunks(&[1, 2]);
        assert_eq!(extracted_texts.len(), 2);
        assert_eq!(
            [
                extracted_texts[0].as_ref().unwrap().trim(),
                extracted_texts[1].as_ref().unwrap().trim()
            ],
            [text1, text2]
        );
    }

    #[test]
    fn extract_text_concatenates_text_from_multiple_pages() {
        use crate::creator::tests::create_document_with_texts;

        let text1 = "Hello world!";
        let text2 = "Ferris is the best!";
        let doc = create_document_with_texts(&[text1, text2]);
        let extracted_text = doc.extract_text(&[1, 2]);
        assert_eq!(extracted_text.unwrap(), format!("{text1}\n{text2}\n"));
    }

    const BOMB_MIB: usize = 1024 * 1024;

    /// A FlateDecode-compressed stream that inflates to `target` zero-bytes, built
    /// by streaming zeros through the compressor so the test process never holds
    /// `target` bytes at once.
    fn flate_bomb(target: usize) -> Vec<u8> {
        use flate2::Compression;
        use flate2::write::ZlibEncoder;
        use std::io::Write;

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
        let zeros = [0u8; 64 * 1024];
        let mut remaining = target;
        while remaining > 0 {
            let n = remaining.min(zeros.len());
            encoder.write_all(&zeros[..n]).unwrap();
            remaining -= n;
        }
        encoder.finish().unwrap()
    }

    /// A `create_document_with_texts` document whose first page's content stream
    /// has been swapped for a FlateDecode bomb that inflates to `target` bytes.
    fn doc_with_page_content_bomb(target: usize) -> crate::Document {
        use crate::{Dictionary, Object, Stream};

        let mut doc = create_document_with_texts(&["Hello"]);
        let page_id = *doc.get_pages().get(&1).expect("page 1 exists");
        let content_id = doc.get_page_contents(page_id)[0];
        let mut dict = Dictionary::new();
        dict.set("Filter", "FlateDecode");
        doc.objects
            .insert(content_id, Object::from(Stream::new(dict, flate_bomb(target))));
        doc
    }

    /// A `create_document_with_texts` document whose font carries a `/ToUnicode`
    /// CMap that is a FlateDecode bomb. Font encodings are resolved (and thus this
    /// stream is decoded) during text extraction, *before* the page content — so
    /// this exercises a decompression vector distinct from page content.
    fn doc_with_tounicode_font_bomb(target: usize) -> crate::Document {
        use crate::{Dictionary, Object, Stream};

        let mut doc = create_document_with_texts(&["Hello"]);
        let font_id = doc
            .objects
            .iter()
            .find(|(_, obj)| obj.as_dict().map(|d| d.has_type(b"Font")).unwrap_or(false))
            .map(|(id, _)| *id)
            .expect("font object exists");

        let mut tounicode_dict = Dictionary::new();
        tounicode_dict.set("Filter", "FlateDecode");
        let tounicode_id = doc.add_object(Object::from(Stream::new(tounicode_dict, flate_bomb(target))));

        let font = doc.get_object_mut(font_id).and_then(Object::as_dict_mut).unwrap();
        font.set("ToUnicode", Object::Reference(tounicode_id));
        doc
    }

    use crate::creator::tests::create_document_with_texts;

    /// Under a generous limit, `extract_text_with_limit` returns exactly the same
    /// text as the unbounded `extract_text`.
    #[test]
    fn extract_text_with_limit_matches_unbounded_under_limit() {
        use crate::creator::tests::create_document_with_texts;

        let text1 = "Hello world!";
        let text2 = "Ferris is the best!";
        let doc = create_document_with_texts(&[text1, text2]);

        let bounded = doc.extract_text_with_limit(&[1, 2], BOMB_MIB).unwrap();
        let unbounded = doc.extract_text(&[1, 2]).unwrap();
        assert_eq!(bounded, unbounded);
    }

    /// A page-content decompression bomb is rejected end-to-end by
    /// `extract_text_with_limit` instead of inflating without bound.
    #[test]
    fn extract_text_with_limit_rejects_page_content_bomb() {
        use crate::{DecompressError, Error};

        let doc = doc_with_page_content_bomb(32 * BOMB_MIB);

        match doc.extract_text_with_limit(&[1], 4 * BOMB_MIB).map(|s| s.len()) {
            Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) => {
                assert_eq!(limit, 4 * BOMB_MIB);
            }
            other => panic!("expected MemoryLimitExceeded, got {other:?}"),
        }
    }

    /// The chunked variant surfaces the same bomb as an `Err` chunk rather than
    /// panicking or silently truncating.
    #[test]
    fn extract_text_chunks_with_limit_surfaces_limit_error() {
        use crate::{DecompressError, Error};

        let doc = doc_with_page_content_bomb(32 * BOMB_MIB);
        let chunks = doc.extract_text_chunks_with_limit(&[1], 4 * BOMB_MIB);

        assert!(
            chunks.iter().any(|chunk| matches!(
                chunk,
                Err(Error::Decompress(DecompressError::MemoryLimitExceeded { .. }))
            )),
            "expected a MemoryLimitExceeded error chunk"
        );
    }

    /// A bomb reached through a font's `/ToUnicode` CMap (decoded during encoding
    /// resolution, before page content) is also bounded by `extract_text_with_limit`
    /// — otherwise the "safe for untrusted input" guarantee has a hole.
    #[test]
    fn extract_text_with_limit_rejects_tounicode_font_bomb() {
        use crate::{DecompressError, Error};

        let doc = doc_with_tounicode_font_bomb(32 * BOMB_MIB);

        match doc.extract_text_with_limit(&[1], 4 * BOMB_MIB).map(|s| s.len()) {
            Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) => {
                assert_eq!(limit, 4 * BOMB_MIB);
            }
            other => panic!("expected MemoryLimitExceeded from ToUnicode font bomb, got {other:?}"),
        }
    }

    #[test]
    fn test_replace_partial_text() {
        use crate::creator::tests::create_document_with_texts;

        let mut doc = create_document_with_texts(&["Hello World! Hello Universe!"]);
        let replacements = doc.replace_partial_text(1, "Hello", "Hi", None).unwrap();
        assert_eq!(replacements, 2); // Should replace both occurrences

        let extracted_text = doc.extract_text(&[1]).unwrap();
        assert!(extracted_text.contains("Hi World! Hi Universe!"));
    }

    /// PDF 1.7 / ISO 32000-1 §9.4.3 — `'` is equivalent to `T* Tj`:
    /// move to the next line and show a string. extract_text should
    /// recover the string operand and emit a line break before it.
    #[test]
    fn extract_text_handles_apostrophe_show_text_op() {
        use crate::Object;
        use crate::content::Operation;
        use crate::creator::tests::create_document_with_operations;

        let doc = create_document_with_operations(vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
            Operation::new("Td", vec![100.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal("first")]),
            Operation::new("'", vec![Object::string_literal("second")]),
            Operation::new("'", vec![Object::string_literal("third")]),
            Operation::new("ET", vec![]),
        ]);

        let text = doc.extract_text(&[1]).unwrap();
        assert!(text.contains("first"), "Tj string lost: {text:?}");
        assert!(text.contains("second"), "first ' string lost: {text:?}");
        assert!(text.contains("third"), "second ' string lost: {text:?}");
    }

    /// PDF 1.7 / ISO 32000-1 §9.4.3 — `"` is equivalent to
    /// `aw Tw ac Tc T* Tj` with operands `[aw, ac, string]`. extract_text
    /// should recover operand index 2 (the string); operands 0 and 1 set
    /// rendering spacing and don't affect the extracted character sequence.
    #[test]
    fn extract_text_handles_quote_show_text_op() {
        use crate::Object;
        use crate::content::Operation;
        use crate::creator::tests::create_document_with_operations;

        let doc = create_document_with_operations(vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
            Operation::new("Td", vec![100.into(), 700.into()]),
            Operation::new("\"", vec![0.into(), 0.into(), Object::string_literal("from-quote-op")]),
            Operation::new("ET", vec![]),
        ]);

        let text = doc.extract_text(&[1]).unwrap();
        assert!(text.contains("from-quote-op"), "\" string operand lost: {text:?}");
    }

    /// PDF 1.7 / ISO 32000-1 §9.4.2 — `T*` moves to the start of the
    /// next line. For text extraction we approximate as `\n`, so a
    /// `Tj T* Tj` sequence should produce two strings separated by a
    /// newline rather than running together.
    #[test]
    fn extract_text_preserves_line_breaks_for_t_star() {
        use crate::Object;
        use crate::content::Operation;
        use crate::creator::tests::create_document_with_operations;

        let doc = create_document_with_operations(vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
            Operation::new("Td", vec![100.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal("line-one")]),
            Operation::new("T*", vec![]),
            Operation::new("Tj", vec![Object::string_literal("line-two")]),
            Operation::new("ET", vec![]),
        ]);

        let text = doc.extract_text(&[1]).unwrap();
        let one = text.find("line-one").expect("line-one missing");
        let two = text.find("line-two").expect("line-two missing");
        assert!(one < two, "order wrong: {text:?}");
        let between = &text[one + "line-one".len()..two];
        assert!(
            between.contains('\n'),
            "T* did not insert a line break between Tj strings: between={between:?}"
        );
    }
}
