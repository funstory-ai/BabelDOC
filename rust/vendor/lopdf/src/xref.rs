use std::collections::BTreeMap;
use std::io::{Result, Write};

#[derive(Debug, Clone)]
pub struct Xref {
    /// Type of Cross-Reference used in the last incremental version.
    /// This method of cross-referencing will also be used when saving the file.
    /// PDFs with Incremental Updates should alway use the same cross-reference type.
    pub cross_reference_type: XrefType,

    /// Entries for indirect object.
    pub entries: BTreeMap<u32, XrefEntry>,

    /// Total number of entries (including free entries), equal to the highest object number plus 1.
    pub size: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum XrefType {
    /// Cross-Reference Streams are supported beginning with PDF 1.5.
    CrossReferenceStream,
    /// Cross-Reference Table is older but still frequently used.
    CrossReferenceTable,
}

#[derive(Debug, Clone)]
pub enum XrefEntry {
    Free, // TODO add generation number
    UnusableFree,
    Normal { offset: u32, generation: u16 },
    Compressed { container: u32, index: u16 },
}

#[derive(Debug, Clone)]
pub struct XrefSection {
    pub starting_id: u32,
    pub entries: Vec<XrefEntry>,
}

impl Xref {
    pub fn new(size: u32, xref_type: XrefType) -> Xref {
        Xref {
            cross_reference_type: xref_type,
            entries: BTreeMap::new(),
            size,
        }
    }

    pub fn get(&self, id: u32) -> Option<&XrefEntry> {
        self.entries.get(&id)
    }

    pub fn insert(&mut self, id: u32, entry: XrefEntry) {
        self.entries.insert(id, entry);
        self.size = self.size.max(id.saturating_add(1));
    }

    /// Combine Xref entries. Only add them if they do not exists already.
    /// Do not replace existing entries.
    pub fn merge(&mut self, xref: Xref) {
        for (id, entry) in xref.entries {
            self.entries.entry(id).or_insert(entry);
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear()
    }

    pub fn max_id(&self) -> u32 {
        match self.entries.keys().max() {
            Some(&id) => id,
            None => 0,
        }
    }
}

impl XrefEntry {
    pub fn is_normal(&self) -> bool {
        matches!(*self, XrefEntry::Normal { .. })
    }

    pub fn is_compressed(&self) -> bool {
        matches!(*self, XrefEntry::Compressed { .. })
    }

    /// Encode entry for use in cross-reference stream
    pub fn encode_for_xref_stream(&self, widths: &[usize; 3]) -> Vec<u8> {
        let mut result = Vec::new();

        match self {
            XrefEntry::Free | XrefEntry::UnusableFree => {
                // Type 0: Free object
                encode_field(0, widths[0], &mut result);
                encode_field(0, widths[1], &mut result); // Next free object
                encode_field(0, widths[2], &mut result); // Generation
            }
            XrefEntry::Normal { offset, generation } => {
                // Type 1: Uncompressed object
                encode_field(1, widths[0], &mut result);
                encode_field(*offset as u64, widths[1], &mut result);
                encode_field(*generation as u64, widths[2], &mut result);
            }
            XrefEntry::Compressed { container, index } => {
                // Type 2: Compressed object
                encode_field(2, widths[0], &mut result);
                encode_field(*container as u64, widths[1], &mut result);
                encode_field(*index as u64, widths[2], &mut result);
            }
        }

        result
    }

    /// Write Entry in Cross Reference Table.
    pub fn write_xref_entry(&self, file: &mut dyn Write) -> Result<()> {
        match self {
            XrefEntry::Normal { offset, generation } => {
                writeln!(file, "{offset:>010} {generation:>05} n ")?;
            }
            XrefEntry::Compressed { container: _, index: _ } => {
                writeln!(file, "{:>010} {:>05} f ", 0, 65535)?;
            }
            XrefEntry::Free => {
                writeln!(file, "{:>010} {:>05} f ", 0, 0)?;
            }
            XrefEntry::UnusableFree => {
                writeln!(file, "{:>010} {:>05} f ", 0, 65535)?;
            }
        }
        Ok(())
    }
}

impl XrefSection {
    pub fn new(starting_id: u32) -> Self {
        XrefSection {
            starting_id,
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: XrefEntry) {
        self.entries.push(entry);
    }

    pub fn add_unusable_free_entry(&mut self) {
        self.add_entry(XrefEntry::UnusableFree);
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Write Section in Cross Reference Table.
    pub fn write_xref_section(&self, file: &mut dyn Write) -> Result<()> {
        if !self.is_empty() {
            // Write section range
            writeln!(file, "{} {}", self.starting_id, self.entries.len())?;
            // Write entries
            for entry in &self.entries {
                entry.write_xref_entry(file)?;
            }
        }
        Ok(())
    }
}

pub use crate::parser_aux::{decode_xref_stream, decode_xref_stream_with_limit};

/// Encode a field value as big-endian bytes with specified width
fn encode_field(value: u64, width: usize, output: &mut Vec<u8>) {
    for i in (0..width).rev() {
        output.push((value >> (i * 8)) as u8);
    }
}

/// Builder for creating cross-reference streams
pub struct XrefStreamBuilder<'a> {
    xref: &'a Xref,
    entries: Vec<(u32, &'a XrefEntry)>,
    widths: [usize; 3],
}

impl<'a> XrefStreamBuilder<'a> {
    /// Create a new builder from an Xref
    pub fn new(xref: &'a Xref) -> Self {
        let entries: Vec<_> = xref.entries.iter().map(|(&id, entry)| (id, entry)).collect();

        Self {
            xref,
            entries,
            widths: [1, 2, 2], // Default widths
        }
    }

    /// Get the number of entries
    pub fn entries_count(&self) -> usize {
        self.entries.len()
    }

    /// Calculate optimal field widths based on the data
    pub fn calculate_optimal_widths(&self) -> [usize; 3] {
        let mut max_offset = 0u64;
        let mut max_gen = 0u16;
        let mut max_container = 0u32;
        let mut max_index = 0u16;

        for (_, entry) in &self.entries {
            match entry {
                XrefEntry::Normal { offset, generation } => {
                    max_offset = max_offset.max(*offset as u64);
                    max_gen = max_gen.max(*generation);
                }
                XrefEntry::Compressed { container, index } => {
                    max_container = max_container.max(*container);
                    max_index = max_index.max(*index);
                }
                _ => {}
            }
        }

        // Calculate bytes needed
        let offset_bytes = bytes_needed(max_offset);
        let gen_bytes = bytes_needed(max_gen as u64);
        let container_bytes = bytes_needed(max_container as u64);
        let index_bytes = bytes_needed(max_index as u64);

        [
            1, // Type field is always 1 byte
            offset_bytes.max(container_bytes),
            gen_bytes.max(index_bytes),
        ]
    }

    /// Build the stream content
    pub fn build_stream_content(&mut self) -> crate::Result<Vec<u8>> {
        self.widths = self.calculate_optimal_widths();
        let mut content = Vec::new();

        // Sort entries by ID
        self.entries.sort_by_key(|(id, _)| *id);

        for (_, entry) in &self.entries {
            let encoded = entry.encode_for_xref_stream(&self.widths);
            content.extend_from_slice(&encoded);
        }

        Ok(content)
    }

    /// Build the Index array for the cross-reference stream
    pub fn build_index_array(&self) -> Vec<crate::Object> {
        use crate::Object;

        let mut index = Vec::new();
        let mut sorted_entries = self.entries.clone();
        sorted_entries.sort_by_key(|(id, _)| *id);

        if sorted_entries.is_empty() {
            return index;
        }

        let mut start = sorted_entries[0].0;
        let mut count = 1;

        for i in 1..sorted_entries.len() {
            if sorted_entries[i].0 == sorted_entries[i - 1].0 + 1 {
                count += 1;
            } else {
                index.push(Object::Integer(start as i64));
                index.push(Object::Integer(count as i64));
                start = sorted_entries[i].0;
                count = 1;
            }
        }

        index.push(Object::Integer(start as i64));
        index.push(Object::Integer(count as i64));

        index
    }

    /// Convert to a Stream object
    pub fn to_stream_object(&mut self) -> crate::Result<crate::Stream> {
        use crate::{Object, Stream, dictionary};

        let content = self.build_stream_content()?;
        let dict = dictionary! {
            "Type" => "XRef",
            "Size" => self.xref.size as i64,
            "W" => vec![
                Object::Integer(self.widths[0] as i64),
                Object::Integer(self.widths[1] as i64),
                Object::Integer(self.widths[2] as i64),
            ],
            "Index" => self.build_index_array(),
            "Filter" => "FlateDecode"
        };

        let mut stream = Stream::new(dict, content);
        stream.compress()?;
        Ok(stream)
    }
}

/// Calculate the minimum number of bytes needed to represent a value
fn bytes_needed(value: u64) -> usize {
    if value == 0 {
        1
    } else {
        (64 - value.leading_zeros()).div_ceil(8) as usize
    }
}
