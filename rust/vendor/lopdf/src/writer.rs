use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path;
use std::vec;

use super::Object::*;
use super::{Dictionary, Document, Object, Stream, StringFormat};
use crate::encryption;
use crate::{IncrementalDocument, xref::*};

impl Document {
    /// Save PDF document to specified file path.
    #[inline]
    pub fn save<P: AsRef<Path>>(&mut self, path: P) -> Result<File> {
        let mut file = BufWriter::new(File::create(path)?);
        self.save_internal(&mut file)?;
        Ok(file.into_inner()?)
    }

    /// Save PDF to arbitrary target
    #[inline]
    pub fn save_to<W: Write>(&mut self, target: &mut W) -> Result<()> {
        self.save_internal(target)
    }

    /// Save PDF with custom options
    ///
    /// Object streams are skipped for an encrypted document, which is written with every
    /// object serialized individually instead. See [`Document::save_modern`].
    pub fn save_with_options<W: Write>(&mut self, target: &mut W, options: crate::SaveOptions) -> Result<()> {
        // Cross-reference streams are independent of object streams: a document can use one
        // without the other. Select the requested type here so the choice applies whichever
        // path writes the body below. Both features arrived in PDF 1.5, so a document that is
        // about to use one is moved up to that version.
        if options.use_xref_streams {
            self.reference_table.cross_reference_type = XrefType::CrossReferenceStream;

            if self.version.as_str() < "1.5" {
                self.version = "1.5".to_string();
            }
        }

        if options.use_object_streams {
            self.save_with_object_streams(target, options)
        } else {
            self.save_internal(target)
        }
    }

    /// Save PDF with modern features (object streams and cross-reference streams)
    ///
    /// An encrypted document is written without object streams. The objects are already
    /// encrypted by the time they reach the writer and the file encryption key is gone,
    /// so an object stream built here could only be written in the clear, contradicting
    /// the document's `/Encrypt` dictionary. The requested cross-reference type is still
    /// used; cross-reference streams are never encrypted.
    pub fn save_modern<W: Write>(&mut self, target: &mut W) -> Result<()> {
        let options = crate::SaveOptions {
            use_object_streams: true,
            use_xref_streams: true,
            ..Default::default()
        };
        self.save_with_options(target, options)
    }

    fn save_internal<W: Write>(&mut self, target: &mut W) -> Result<()> {
        let mut target = CountingWrite {
            inner: target,
            bytes_written: 0,
        };

        let mut xref = Xref::new(self.max_id + 1, self.reference_table.cross_reference_type);
        writeln!(target, "%PDF-{}", self.version)?;

        Writer::write_binary_mark(&mut target, &self.binary_mark)?;

        for (&(id, generation), object) in &self.objects {
            if object
                .type_name()
                .map(|name| [b"ObjStm".as_slice(), b"XRef".as_slice(), b"Linearized".as_slice()].contains(&name))
                .ok()
                != Some(true)
            {
                Writer::write_indirect_object(&mut target, id, generation, object, &mut xref)?;
            }
        }

        let xref_start = target.bytes_written;

        // Pick right cross reference stream.
        match xref.cross_reference_type {
            XrefType::CrossReferenceTable => {
                Writer::write_xref(&mut target, &xref)?;
                self.write_trailer(&mut target)?;
            }
            XrefType::CrossReferenceStream => {
                // Cross Reference Stream instead of XRef and Trailer
                self.write_cross_reference_stream(&mut target, &mut xref, xref_start as u32)?;
            }
        }
        // Write `startxref` part of trailer
        write!(target, "\nstartxref\n{xref_start}\n%%EOF")?;

        Ok(())
    }

    /// Save PDF with object streams enabled
    fn save_with_object_streams<W: Write>(&mut self, target: &mut W, options: crate::SaveOptions) -> Result<()> {
        use crate::ObjectStream;
        use std::collections::HashMap;

        // Ensure PDF version is at least 1.5 (required for object streams)
        if self.version.as_str() < "1.5" {
            self.version = "1.5".to_string();
        }

        // Object streams are built here, while serializing, but the document's objects were
        // already encrypted by `Document::encrypt`, which drops the file encryption key once
        // it is done. There is nothing left to encrypt a new stream with, so it would go out
        // in the clear while the `/Encrypt` dictionary claims every stream is encrypted, and
        // the strings packed into it would stay encrypted a second time: an object stream is
        // itself the unit of encryption, and strings inside one shall not be encrypted
        // separately. Write the objects out individually instead, as `save` does. The
        // cross-reference type chosen in `save_with_options` still applies.
        if self.is_encrypted() {
            return self.save_internal(target);
        }

        let mut target = CountingWrite {
            inner: target,
            bytes_written: 0,
        };

        let mut xref = Xref::new(self.max_id + 1, self.reference_table.cross_reference_type);
        writeln!(target, "%PDF-{}", self.version)?;
        Writer::write_binary_mark(&mut target, &self.binary_mark)?;

        // Organize objects into streams
        let mut object_streams: Vec<crate::ObjectStream> = Vec::new();
        let mut objects_to_write_directly = Vec::new();
        let mut object_to_stream_map = HashMap::new();

        // Categorize objects
        for (&(id, generation), object) in &self.objects {
            // Skip existing object streams - we'll create new ones
            if let Object::Stream(stream) = object
                && let Ok(type_obj) = stream.dict.get(b"Type")
                && let Ok(type_name) = type_obj.as_name()
                && type_name == b"ObjStm"
            {
                continue; // Skip existing object streams
            }

            if generation == 0 && ObjectStream::can_be_compressed((id, generation), object, self) {
                // Object can be compressed
                // Find or create an object stream for it
                let stream_index = object_streams.len().saturating_sub(1);

                if object_streams.is_empty()
                    || object_streams[stream_index].object_count()
                        >= options.object_stream_config.max_objects_per_stream
                {
                    // Create new object stream
                    let new_stream = ObjectStream::builder()
                        .max_objects(options.object_stream_config.max_objects_per_stream)
                        .compression_level(options.object_stream_config.compression_level)
                        .build();
                    object_streams.push(new_stream);
                }

                let stream_index = object_streams.len() - 1;
                object_streams[stream_index]
                    .add_object((id, generation), object.clone())
                    .ok();
                object_to_stream_map.insert((id, generation), stream_index);
            } else {
                // Object must be written directly
                objects_to_write_directly.push(((id, generation), object));
            }
        }

        // Write direct objects first
        for ((id, generation), object) in objects_to_write_directly {
            Writer::write_indirect_object(&mut target, id, generation, object, &mut xref)?;
        }

        // Write object streams
        let mut stream_count = 0;
        for obj_stream in object_streams.into_iter() {
            let stream_id = self.max_id + 1 + stream_count;
            let stream_obj = obj_stream.to_stream_object().map_err(std::io::Error::other)?;

            // Record compressed objects in xref
            // Must use the same sort order as build_stream_content()
            let mut sorted_objects: Vec<_> = obj_stream.objects.keys().cloned().collect();
            sorted_objects.sort_by_key(|id| *id);
            for (index_in_stream, (obj_id, _gen)) in sorted_objects.iter().enumerate() {
                xref.insert(
                    *obj_id,
                    XrefEntry::Compressed {
                        container: stream_id,
                        index: index_in_stream as u16,
                    },
                );
            }

            // Write the object stream
            Writer::write_indirect_object(&mut target, stream_id, 0, &Object::from(stream_obj), &mut xref)?;
            stream_count += 1;
        }

        // Update max_id to account for object streams
        self.max_id += stream_count;

        let xref_start = target.bytes_written;

        // Write cross-reference
        match xref.cross_reference_type {
            XrefType::CrossReferenceTable => {
                Writer::write_xref(&mut target, &xref)?;
                self.write_trailer(&mut target)?;
            }
            XrefType::CrossReferenceStream => {
                self.write_cross_reference_stream(&mut target, &mut xref, xref_start as u32)?;
            }
        }

        write!(target, "\nstartxref\n{xref_start}\n%%EOF")?;
        Ok(())
    }

    /// Write the Cross Reference Stream.
    ///
    /// Insert an `Object` to the end of the PDF (not visible when inspecting `Document`).
    /// Note: This is different from the "Cross Reference Table".
    fn write_cross_reference_stream<W: Write>(
        &mut self, file: &mut CountingWrite<&mut W>, xref: &mut Xref, xref_start: u32,
    ) -> Result<()> {
        // Increment max_id to account for CRS.
        self.max_id += 1;
        let new_obj_id_for_crs = self.max_id;
        xref.insert(
            new_obj_id_for_crs,
            XrefEntry::Normal {
                offset: xref_start,
                generation: 0,
            },
        );
        self.trailer.set("Type", Name(b"XRef".to_vec()));
        // Update `max_id` in trailer
        self.trailer.set("Size", i64::from(self.max_id + 1));
        // Set the size of each entry in bytes (default for PDFs is `[1 2 1]`)
        // In our case we use `[u8, u32, u16]` for each entry
        // to keep things simple and working at all times.
        self.trailer.set("W", Array(vec![Integer(1), Integer(4), Integer(2)]));
        // Note that `ASCIIHexDecode` does not work correctly,
        // but is still useful for debugging sometimes.
        let filter = XRefStreamFilter::None;
        let (stream, stream_length, indexes) = Writer::create_xref_steam(xref, filter)?;
        self.trailer.set("Index", indexes);

        if filter == XRefStreamFilter::ASCIIHexDecode {
            self.trailer.set("Filter", Name(b"ASCIIHexDecode".to_vec()));
        } else {
            self.trailer.remove(b"Filter");
        }

        self.trailer.set("Length", stream_length as i64);

        let trailer = &self.trailer;
        let cross_reference_stream = Object::from(Stream {
            dict: trailer.clone(),
            allows_compression: true,
            content: stream.into(),
            start_position: None,
        });
        // Insert Cross Reference Stream as an `Object` to the end of the PDF.
        // The `Object` is not added to `Document` because it is generated every time you save.
        Writer::write_indirect_object(file, new_obj_id_for_crs, 0, &cross_reference_stream, xref)?;

        Ok(())
    }

    fn write_trailer(&mut self, file: &mut dyn Write) -> Result<()> {
        self.trailer.set("Size", i64::from(self.max_id + 1));
        file.write_all(b"trailer\n")?;
        Writer::write_dictionary(file, &self.trailer)?;
        Ok(())
    }
}

impl IncrementalDocument {
    /// Save PDF document to specified file path.
    ///
    /// The `check_incremental_save_supported` guard is invoked before
    /// `File::create` so an unsupported input (e.g. a still-encrypted
    /// previous revision) does not truncate a pre-existing file at `path`.
    #[inline]
    pub fn save<P: AsRef<Path>>(&mut self, path: P) -> Result<File> {
        self.check_incremental_save_supported()?;
        let mut file = BufWriter::new(File::create(path)?);
        self.save_internal(&mut file)?;
        Ok(file.into_inner()?)
    }

    /// Save PDF to arbitrary target
    #[inline]
    pub fn save_to<W: Write>(&mut self, target: &mut W) -> Result<()> {
        self.save_internal(target)
    }

    /// Reject the two cases we still cannot handle: a document that arrived
    /// still-encrypted (no password was supplied), and the inconsistent case
    /// of `encryption_state` set but `encrypt_object_id` missing (which should
    /// not occur in practice — `decrypt_raw` records the id — but we guard
    /// against it defensively).
    fn check_incremental_save_supported(&self) -> Result<()> {
        let prev = self.get_prev_documents();
        if prev.is_encrypted() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "incremental update of a still-encrypted PDF is not supported: \
                 call `Document::decrypt` on the previous revision first \
                 (see https://github.com/J-F-Liu/lopdf/issues/520)",
            ));
        }
        if let Some(state) = prev.encryption_state.as_ref()
            && state.encrypt_object_id().is_none()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "cannot incrementally save this decrypted document: \
                 the /Encrypt object id was not recorded during decryption",
            ));
        }
        Ok(())
    }

    fn save_internal<W: Write>(&mut self, target: &mut W) -> Result<()> {
        self.check_incremental_save_supported()?;

        // If the previous revision was encrypted (and successfully decrypted),
        // we need to re-encrypt every appended object with the same encryption
        // state and restore the trailer's `/Encrypt` reference.
        //
        // Cloning `EncryptionState` (small, mostly `Vec<u8>`) avoids borrow
        // conflicts between `&self.prev_documents` and `&mut self.new_document`.
        let encryption_state = self.get_prev_documents().encryption_state.as_ref().cloned();

        let mut target = CountingWrite {
            inner: target,
            bytes_written: 0,
        };

        // Write previous document versions.
        let prev_document_bytes = self.get_prev_documents_bytes();
        target.inner.write_all(prev_document_bytes)?;
        target.bytes_written += prev_document_bytes.len();

        // Write/Append new document version.
        let mut xref = Xref::new(
            self.new_document.max_id + 1,
            self.get_prev_documents().reference_table.cross_reference_type,
        );

        if let Some(last_byte) = prev_document_bytes.last()
            && *last_byte != b'\n'
        {
            // Add a newline if it was not already present
            writeln!(target)?;
        }

        // No file header and no binary marker here. An incremental update is
        // defined (ISO 32000-1, 7.5.6) as the original file followed by the
        // changed objects, a cross-reference section and a trailer — the
        // header belongs to the file, which the previous revision already
        // carries. Emitting a second "%PDF-x.y" makes the appended region
        // look like the start of another document to anything that locates a
        // PDF by scanning for the header, and because `new_document`'s
        // version defaults to 1.4 it also understated the format of every
        // file built on a later version.

        // Write each newly added indirect object. When the document is
        // encrypted, each object is cloned first and the clone is encrypted;
        // the in-memory objects are left as plaintext so that further edits
        // and repeated saves do not double-encrypt.
        for (&(id, generation), object) in &self.new_document.objects {
            if object
                .type_name()
                .map(|name| [b"ObjStm".as_slice(), b"XRef".as_slice(), b"Linearized".as_slice()].contains(&name))
                .ok()
                != Some(true)
            {
                if let Some(state) = encryption_state.as_ref() {
                    let mut encrypted = object.clone();
                    encryption::encrypt_object(state, (id, generation), &mut encrypted)
                        .map_err(std::io::Error::other)?;
                    Writer::write_indirect_object(&mut target, id, generation, &encrypted, &mut xref)?;
                } else {
                    Writer::write_indirect_object(&mut target, id, generation, object, &mut xref)?;
                }
            }
        }

        // For an encrypted document, install a modified copy of the trailer
        // that restores the /Encrypt reference. We swap it in temporarily so
        // that `write_trailer` / `write_cross_reference_stream` — which
        // already mutate the trailer to update Size/W/Length/etc — operate
        // on the modified copy, and swap it back afterwards so that the
        // in-memory `new_document.trailer` stays clean for subsequent saves.
        let saved_trailer = if let Some(state) = encryption_state.as_ref() {
            let encrypt_id = state
                .encrypt_object_id()
                .expect("encrypt_object_id presence checked by check_incremental_save_supported");
            let mut modified = self.new_document.trailer.clone();
            modified.set(b"Encrypt", Object::Reference(encrypt_id));
            Some(std::mem::replace(&mut self.new_document.trailer, modified))
        } else {
            None
        };

        let xref_start = target.bytes_written;

        // Pick right cross reference stream.
        let write_result: Result<()> = (|| {
            match xref.cross_reference_type {
                XrefType::CrossReferenceTable => {
                    Writer::write_xref(&mut target, &xref)?;
                    self.new_document.write_trailer(&mut target)?;
                }
                XrefType::CrossReferenceStream => {
                    // Cross Reference Stream instead of XRef and Trailer
                    self.new_document
                        .write_cross_reference_stream(&mut target, &mut xref, xref_start as u32)?;
                }
            }
            // Write `startxref` part of trailer
            write!(target, "\nstartxref\n{xref_start}\n%%EOF")?;
            Ok(())
        })();

        // Restore the original in-memory trailer even if writing failed.
        if let Some(saved) = saved_trailer {
            self.new_document.trailer = saved;
        }

        write_result
    }
}

pub struct Writer;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum XRefStreamFilter {
    ASCIIHexDecode,
    _FlateDecode, //this is generally a Zlib compressed Stream.
    None,
}

impl Writer {
    fn need_separator(object: &Object) -> bool {
        matches!(*object, Null | Boolean(_) | Integer(_) | Real(_) | Reference(_))
    }

    fn need_end_separator(object: &Object) -> bool {
        matches!(
            *object,
            Null | Boolean(_) | Integer(_) | Real(_) | Name(_) | Reference(_) | Object::Stream(_)
        )
    }

    /// Write Cross Reference Table.
    ///
    /// Note: This is different from a "Cross Reference Stream".
    fn write_xref(file: &mut dyn Write, xref: &Xref) -> Result<()> {
        writeln!(file, "xref")?;

        let mut xref_section = XrefSection::new(0);
        // Add first (0) entry
        xref_section.add_unusable_free_entry();

        // Iterate over the actual highest entry instead of `xref.size`:
        // `size` is fixed before object streams (and the xref stream itself)
        // are appended, so entries past it would never reach the table.
        for obj_id in 1..=xref.max_id() {
            if let Some(entry) = xref.get(obj_id) {
                // A section starts at the first *present* id; starting it at
                // a missing id would shift every subsequent entry by one.
                if xref_section.is_empty() {
                    xref_section = XrefSection::new(obj_id);
                }
                match *entry {
                    XrefEntry::Normal { offset, generation } => {
                        // Add entry
                        xref_section.add_entry(XrefEntry::Normal { offset, generation });
                    }
                    XrefEntry::Compressed { container: _, index: _ } => {
                        xref_section.add_unusable_free_entry();
                    }
                    XrefEntry::Free => {
                        xref_section.add_entry(XrefEntry::Free);
                    }
                    XrefEntry::UnusableFree => {
                        xref_section.add_unusable_free_entry();
                    }
                }
            } else {
                // Skip over `obj_id`, but finish section if not empty.
                if !xref_section.is_empty() {
                    xref_section.write_xref_section(file)?;
                    xref_section = XrefSection::new(0);
                }
            }
        }
        // Print last section
        if !xref_section.is_empty() {
            xref_section.write_xref_section(file)?;
        }
        Ok(())
    }

    /// Create stream for Cross reference stream.
    fn create_xref_steam(xref: &Xref, filter: XRefStreamFilter) -> Result<(Vec<u8>, usize, Object)> {
        let mut xref_sections = Vec::new();
        let mut xref_section = XrefSection::new(0);

        // Iterate over the actual highest entry instead of `xref.size`:
        // `size` is fixed before object streams (and the xref stream itself)
        // are appended, so entries past it would never reach the stream.
        for obj_id in 1..=xref.max_id() {
            if let Some(entry) = xref.get(obj_id) {
                // A section starts at the first *present* id; starting it at
                // a missing id would shift every subsequent entry by one.
                if xref_section.is_empty() {
                    xref_section = XrefSection::new(obj_id);
                }
                xref_section.add_entry(entry.clone());
            } else {
                // Skip over but finish section if not empty
                if !xref_section.is_empty() {
                    xref_sections.push(xref_section);
                    xref_section = XrefSection::new(0);
                }
            }
        }
        // Print last section
        if !xref_section.is_empty() {
            xref_sections.push(xref_section);
        }

        let mut xref_stream = Vec::new();
        let mut xref_index = Vec::new();

        for section in xref_sections {
            // Add indexes to list
            xref_index.push(Integer(section.starting_id as i64));
            xref_index.push(Integer(section.entries.len() as i64));
            // Add entries to stream
            for (obj_id, entry) in (section.starting_id..).zip(section.entries) {
                match entry {
                    XrefEntry::Free => {
                        // Type 0
                        xref_stream.push(0);
                        xref_stream.extend(obj_id.to_be_bytes());
                        xref_stream.extend(vec![0, 0]); // TODO add generation number
                    }
                    XrefEntry::UnusableFree => {
                        // Type 0
                        xref_stream.push(0);
                        xref_stream.extend(obj_id.to_be_bytes());
                        xref_stream.extend(65535_u16.to_be_bytes());
                    }
                    XrefEntry::Normal { offset, generation } => {
                        // Type 1
                        xref_stream.push(1);
                        xref_stream.extend(offset.to_be_bytes());
                        xref_stream.extend(generation.to_be_bytes());
                    }
                    XrefEntry::Compressed { container, index } => {
                        // Type 2
                        xref_stream.push(2);
                        xref_stream.extend(container.to_be_bytes());
                        xref_stream.extend(index.to_be_bytes());
                    }
                }
            }
        }

        // The end of line character should not be counted, added later.
        let stream_length = xref_stream.len();

        if filter == XRefStreamFilter::ASCIIHexDecode {
            xref_stream = xref_stream
                .iter()
                .flat_map(|c| format!("{c:02X}").as_bytes().to_vec())
                .collect::<Vec<u8>>();
        }

        Ok((xref_stream, stream_length, Array(xref_index)))
    }

    fn write_indirect_object<W: Write>(
        file: &mut CountingWrite<&mut W>, id: u32, generation: u16, object: &Object, xref: &mut Xref,
    ) -> Result<()> {
        let offset = file.bytes_written as u32;
        xref.insert(id, XrefEntry::Normal { offset, generation });
        write!(
            file,
            "{} {} obj\n{}",
            id,
            generation,
            if Writer::need_separator(object) { " " } else { "" }
        )?;
        Writer::write_object(file, object)?;
        writeln!(
            file,
            "{}\nendobj",
            if Writer::need_end_separator(object) { " " } else { "" }
        )?;
        Ok(())
    }

    pub fn write_object(file: &mut dyn Write, object: &Object) -> Result<()> {
        match object {
            Null => file.write_all(b"null"),
            Boolean(value) => {
                if *value {
                    file.write_all(b"true")
                } else {
                    file.write_all(b"false")
                }
            }
            Integer(value) => {
                let mut buf = itoa::Buffer::new();
                file.write_all(buf.format(*value).as_bytes())
            }
            Real(value) => write!(file, "{value}"),
            Name(name) => Writer::write_name(file, name),
            String(text, format) => Writer::write_string(file, text, format),
            Array(array) => Writer::write_array(file, array),
            Object::Dictionary(dict) => Writer::write_dictionary(file, dict),
            Object::Stream(stream) => Writer::write_stream(file, stream),
            Reference(id) => write!(file, "{} {} R", id.0, id.1),
        }
    }

    fn write_name(file: &mut dyn Write, name: &[u8]) -> Result<()> {
        file.write_all(b"/")?;
        for &byte in name {
            // white-space and delimiter chars are encoded to # sequences
            // also encode bytes outside of the range 33 (!) to 126 (~)
            if b" \t\n\r\x0C()<>[]{}/%#".contains(&byte) || !(33..=126).contains(&byte) {
                write!(file, "#{byte:02X}")?;
            } else {
                file.write_all(&[byte])?;
            }
        }
        Ok(())
    }

    fn write_string(file: &mut dyn Write, text: &[u8], format: &StringFormat) -> Result<()> {
        match *format {
            // Within a Literal string, backslash (\) and unbalanced parentheses should be escaped.
            // This rule apply to each individual byte in a string object,
            // whether the string is interpreted as single-byte or multiple-byte character codes.
            // If an end-of-line marker appears within a literal string without a preceding backslash, the result is
            // equivalent to \n. So \r also need be escaped.
            StringFormat::Literal => {
                let mut escape_indice = Vec::new();
                let mut parentheses = Vec::new();
                for (index, &byte) in text.iter().enumerate() {
                    match byte {
                        b'(' => parentheses.push(index),
                        b')' => {
                            if !parentheses.is_empty() {
                                parentheses.pop();
                            } else {
                                escape_indice.push(index);
                            }
                        }
                        b'\\' | b'\r' => escape_indice.push(index),
                        _ => continue,
                    }
                }
                escape_indice.append(&mut parentheses);

                file.write_all(b"(")?;
                if !escape_indice.is_empty() {
                    for (index, &byte) in text.iter().enumerate() {
                        if escape_indice.contains(&index) {
                            file.write_all(b"\\")?;
                            file.write_all(&[if byte == b'\r' { b'r' } else { byte }])?;
                        } else {
                            file.write_all(&[byte])?;
                        }
                    }
                } else {
                    file.write_all(text)?;
                }
                file.write_all(b")")?;
            }
            StringFormat::Hexadecimal => {
                file.write_all(b"<")?;
                for &byte in text {
                    write!(file, "{byte:02X}")?;
                }
                file.write_all(b">")?;
            }
        }
        Ok(())
    }

    fn write_array(file: &mut dyn Write, array: &[Object]) -> Result<()> {
        file.write_all(b"[")?;
        let mut first = true;
        for object in array {
            if first {
                first = false;
            } else if Writer::need_separator(object) {
                file.write_all(b" ")?;
            }
            Writer::write_object(file, object)?;
        }
        file.write_all(b"]")?;
        Ok(())
    }

    fn write_dictionary(file: &mut dyn Write, dictionary: &Dictionary) -> Result<()> {
        file.write_all(b"<<")?;
        for (key, value) in dictionary {
            Writer::write_name(file, key)?;
            if Writer::need_separator(value) {
                file.write_all(b" ")?;
            }
            Writer::write_object(file, value)?;
        }
        file.write_all(b">>")?;
        Ok(())
    }

    fn write_stream(file: &mut dyn Write, stream: &Stream) -> Result<()> {
        Writer::write_dictionary(file, &stream.dict)?;
        file.write_all(b"stream\n")?;
        file.write_all(&stream.content)?;
        file.write_all(b"\nendstream")?;
        Ok(())
    }

    /// Write Binary mark as follows: %{binary_mark[4]}\n -> %Çì¢ or Hex(%25 c3 87 c3 ac)
    ///
    /// Note: Specified in  ISO 19005-2:2011, ISO 19005-3:2012
    /// headerByte1 > 127 && headerByte2 > 127 && headerByte3 > 127 && headerByte4 > 127
    fn write_binary_mark(file: &mut dyn Write, binary_mark: &[u8]) -> Result<()> {
        if binary_mark.iter().all(|&byte| byte >= 128) {
            file.write_all(b"%")?;
            file.write_all(binary_mark)?;
            file.write_all(b"\n")?;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid binary mark",
            ));
        }

        Ok(())
    }
}

pub struct CountingWrite<W: Write> {
    inner: W,
    bytes_written: usize,
}

impl<W: Write> Write for CountingWrite<W> {
    #[inline]
    fn write(&mut self, buffer: &[u8]) -> Result<usize> {
        let result = self.inner.write(buffer);
        if let Ok(bytes) = result {
            self.bytes_written += bytes;
        }
        result
    }

    #[inline]
    fn write_all(&mut self, buffer: &[u8]) -> Result<()> {
        self.bytes_written += buffer.len();
        // If this returns `Err` we can’t know how many bytes were actually written (if any)
        // but that doesn’t matter since we’re gonna abort the entire PDF generation anyway.
        self.inner.write_all(buffer)
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }
}

#[test]
fn save_document() {
    let mut doc = Document::with_version("1.5");
    doc.objects.insert((1, 0), Null);
    doc.objects.insert((2, 0), Boolean(true));
    doc.objects.insert((3, 0), Integer(3));
    doc.objects.insert((4, 0), Real(0.5));
    doc.objects
        .insert((5, 0), String("text((\r)".as_bytes().to_vec(), StringFormat::Literal));
    doc.objects.insert(
        (6, 0),
        String("text((\r)".as_bytes().to_vec(), StringFormat::Hexadecimal),
    );
    doc.objects.insert((7, 0), Name(b"name \t".to_vec()));
    doc.objects.insert((8, 0), Reference((1, 0)));
    doc.objects
        .insert((9, 2), Array(vec![Integer(1), Integer(2), Integer(3)]));
    doc.objects
        .insert((11, 0), Stream(Stream::new(Dictionary::new(), vec![0x41, 0x42, 0x43])));
    let mut dict = Dictionary::new();
    dict.set("A", Null);
    dict.set("B", false);
    dict.set("C", Name(b"name".to_vec()));
    doc.objects.insert((12, 0), Object::Dictionary(dict));
    doc.max_id = 12;

    // Create temporary folder to store file.
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("test_0_save.pdf");
    doc.save(&file_path).unwrap();
    // Check if file was created.
    assert!(file_path.exists());
    // Check if path is file
    assert!(file_path.is_file());
    // Check if the file is above 400 bytes (should be about 610 bytes)
    assert!(file_path.metadata().unwrap().len() > 400);
}
